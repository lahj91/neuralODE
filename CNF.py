import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint_adjoint as odeint
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# 인자 파서 설정
parser = argparse.ArgumentParser('CNF for Two Moons')
parser.add_argument('--adjoint', type=eval, default=True, choices=[True, False])
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args([]) # Jupyter/Colab 환경을 위해 빈 리스트로 초기화

# Adjoint 방식을 사용할지 여부 설정
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


def trace_df_dz(f, z):
    sum_diag = 0.0
    for i in range(z.shape[1]):
        e = torch.zeros_like(z)
        e[:, i] = 1.0
        # f(z)와 e의 내적을 z에 대해 미분
        grad_output = torch.autograd.grad(f(z), z, e, create_graph=True)[0]
        # 결과의 i번째 성분이 대각합의 i번째 원소에 해당
        sum_diag += grad_output[:, i]
    return sum_diag.contiguous()

class ODEFunc(nn.Module):
    """
    ODE의 동역학(dynamics) dz/dt = f(t, z)를 정의하는 신경망
    이 신경망은 또한 로그 확률의 변화를 계산하기 위해 야코비안의 대각합을 추정
    """
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.n_evals = 0

    def forward(self, t, z_and_logp):
        """
        z와 logp_z를 합친 텐서를 입력받아,
        dz/dt와 d(log p(z(t)))/dt를 계산하여 반환합니다.
        """
        # solver에서 network를 몇 번 호출했는지 확인 
        self.n_evals += 1
        
        # torch.set_grad_enabled(True) 컨텍스트 내에서 그래디언트 계산을 활성화
        with torch.set_grad_enabled(True):
            z, _ = z_and_logp[:, :2], z_and_logp[:, 2:]
            
            # requires_grad_(True)를 호출하여 z에 대한 그래디언트 계산을 활성화
            z.requires_grad_(True)

            # dz/dt 계산
            dz_dt = self.fc3(self.elu(self.fc2(self.elu(self.fc1(z)))))
            
            # d(log p(z(t)))/dt = -tr(df/dz) 계산
            # df/dz는 dz_dt를 z로 미분한 야코비안 행렬
            dlogp_z_dt = -trace_df_dz(lambda x: self.fc3(self.elu(self.fc2(self.elu(self.fc1(x))))), z)

        return torch.cat([dz_dt, dlogp_z_dt.view(-1, 1)], dim=1)


class CNF(nn.Module):
    """Continuous Normalizing Flow 모델"""
    def __init__(self, odefunc):
        super(CNF, self).__init__()
        self.odefunc = odefunc

    def forward(self, z, logp_z, integration_times=None, reverse=False):
        if integration_times is None:
            integration_times = torch.tensor([0.0, 1.0]).to(z)
        
        if reverse:
            integration_times = _flip(integration_times, 0)
            
        # 초기 상태: z와 logp_z를 결합
        z_and_logp = torch.cat([z, logp_z], dim=1)
        
        # ODE 풀기
        z_and_logp_t = odeint(
            self.odefunc,
            z_and_logp,
            integration_times,
            atol=1e-5,
            rtol=1e-5,
            method='dopri5',
        )
        
        # 결과 분리
        z_t, logp_z_t = z_and_logp_t[-1][:, :2], z_and_logp_t[-1][:, 2:]
        
        return z_t, logp_z_t


def _flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def get_batch(num_samples):
    """Two Moons 데이터셋에서 샘플을 생성합니다."""
    points, _ = make_moons(n_samples=num_samples, noise=0.06)
    x = torch.tensor(points).type(torch.float32)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32)
    return x, logp_diff_t1


def main():
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    
    # 모델 정의
    odefunc = ODEFunc(args.hidden_dim)
    model = CNF(odefunc).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 기본 분포 (Standard Normal)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
    )
    
    # 시각화 준비
    if args.viz:
        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        fig = plt.figure(figsize=(12, 4), dpi=150)
        
    print("학습 시작...")
    start_time = time.time()
    
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        
        # 데이터 샘플링
        x, logp_diff_t1 = get_batch(args.num_samples)
        x = x.to(device)
        logp_diff_t1 = logp_diff_t1.to(device)
        
        # x -> z 변환 및 log-determinant 계산
        z_t, logp_diff_t = model(x, logp_diff_t1)
        
        # z의 로그 확률 계산 (기본 분포 하에서)
        logp_z = p_z0.log_prob(z_t).to(device).view(-1, 1)
        
        # 최종 로그 확률 = log p(z) + log |det(dz/dx)|
        logp_x = logp_z - logp_diff_t
        
        # Loss = Negative Log-Likelihood
        loss = -logp_x.mean()
        
        loss.backward()
        optimizer.step()
        
        if itr % 100 == 0:
            elapsed_time = time.time() - start_time
            print(f'Iter: {itr}, Loss: {loss.item():.4f}, Time: {elapsed_time:.2f}s')

            if args.viz:
                visualize(fig, x, model, p_z0, itr)

    print("학습 완료.")
    if args.viz:
        plt.show()

def visualize(fig, x, model, p_z0, itr):
    """학습 과정 및 결과를 시각화합니다."""
    device = x.device
    fig.clf()

    # 1. 원본 데이터 분포
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_title(f'p_data(x) at iter {itr}')
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(-1.5, 2.5)
    ax1.set_ylim(-1, 1.5)
    x_np = x.cpu().numpy()
    ax1.scatter(x_np[:, 0], x_np[:, 1], c='black', s=10, alpha=0.5)

    # 2. 모델이 학습한 변환 (x -> z)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title('p_z(z)')
    ax2.set_aspect('equal')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    
    with torch.no_grad():
        z, _ = model(x, torch.zeros(x.shape[0], 1).to(device))
        z_np = z.cpu().numpy()
        ax2.scatter(z_np[:, 0], z_np[:, 1], c='black', s=10, alpha=0.5)

    # 3. 생성된 샘플 (z -> x, 역변환)
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title('Generated Samples')
    ax3.set_aspect('equal')
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlim(-1.5, 2.5)
    ax3.set_ylim(-1, 1.5)
    
    with torch.no_grad():
        # 기본 분포에서 샘플링
        z_samples = p_z0.sample([args.num_samples]).to(device)
        logp_z_samples = p_z0.log_prob(z_samples).view(-1, 1)
        
        # 역변환으로 데이터 생성
        x_generated, _ = model(z_samples, logp_z_samples, reverse=True)
        x_gen_np = x_generated.cpu().numpy()
        ax3.scatter(x_gen_np[:, 0], x_gen_np[:, 1], c='black', s=10, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, f"cnf-two-moons-{itr:04d}.png"), dpi=150)
    plt.draw()
    plt.pause(0.1)


if __name__ == '__main__':
    # 시각화를 활성화하여 실행
    args.viz = True
    args.niters = 10000
    args.lr = 1e-3
    args.num_samples = 1024
    
    main()
