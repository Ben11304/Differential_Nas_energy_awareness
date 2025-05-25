import torch
import numpy as np
import torch.nn as nn

def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])

class Architect(object):
    def __init__(self, model, args, device='cuda'):
        """
        Thêm tham số device để bảo đảm ta có thể chạy trên CPU hoặc GPU tùy ý.
        """
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self._device = device

        # Optimizer cho các kiến trúc alpha (arch_parameters)
        self.optimizer = torch.optim.Adam(
            self.model.arch_parameters(),
            lr=args.arch_learning_rate,
            betas=(0.5, 0.999),
            weight_decay=args.arch_weight_decay
        )
        self.max_seen_mac = torch.tensor(1e-6, device=args.device)
        self.max_seen_loss = torch.tensor(1e-6, device=args.device)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        """
        Tạo một “unrolled model” tức là mô hình giả lập sau một bước update tham số (theta)
        nhưng vẫn giữ alpha cũ. Dùng trong unrolled optimization (mục đích: tính gradient alpha).
        """
        # Tính loss trên batch train
        loss = self.model._loss(input, target)

        # Nối tất cả parameters của model (trừ alpha) thành vector theta
        theta = _concat(self.model.parameters()).detach()

        # Lấy buffer momentum, nếu chưa có thì tạo 0
        try:
            moment = _concat(
                network_optimizer.state[v]['momentum_buffer'] 
                for v in self.model.parameters()
            ).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        # Tính gradient dtheta wrt (các params) + weight_decay * theta
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters()))
        dtheta = dtheta + self.network_weight_decay * theta

        # Tạo unrolled_model với theta mới = theta - eta * (moment + dtheta)
        unrolled_theta = theta - eta * (moment + dtheta)
        unrolled_model = self._construct_model_from_theta(unrolled_theta)
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, 
             eta, network_optimizer, unrolled,rate):
        """
        Bước update alpha (arch_parameters):
          - unrolled=True: dùng unrolled optimization (1 step SGD ảo).
          - unrolled=False: dùng “simple” optimization.
        """
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                input_train, target_train,
                input_valid, target_valid,
                eta, network_optimizer
            )
        else:
            self._backward_step(input_valid, target_valid,rate)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid, rate=[0.9,0.1]):
        """
        Tính gradient alpha đơn giản: loss trên batch valid wrt alpha
        """
        alphaa = 0.9
        output,energy= self.model(input_valid)
        loss_task=self.model._loss(input_valid, target_valid)
        loss_energy=energy
        self.max_seen_mac = torch.max(self.max_seen_mac * alphaa, loss_energy.detach())
        self.max_seen_loss = torch.max(self.max_seen_loss * alphaa, loss_task.detach())
        energy_normalized = loss_energy / self.max_seen_mac
        loss_task_normalized = loss_task/self.max_seen_loss
        
        # output,energy= self.model(input_valid)
        loss=rate[0]*loss_task_normalized+rate[1]*energy_normalized
        # print(f"-----------------{loss_task}------------")
        # print(f"-----------------{loss_energy}------------")
        # loss=energy_normalized
        # loss=loss_task_normalized
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, 
                                input_valid, target_valid, eta, network_optimizer):
        """
        Tính gradient alpha theo kiểu unrolled:
          1) Tạo unrolled_model (theta đã update 1 bước)
          2) Tính loss(valid) trên unrolled_model
          3) backprop lên alpha (dalpha)
          4) Cộng thêm term “hessian_vector_product” (công thức trong DARTS)
        """
        # 1) Tạo unrolled_model
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)

        # 2) Loss trên batch valid
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)

        # 3) Backprop ra grad wrt alpha của unrolled_model
        unrolled_loss.backward()
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]

        # vector = grad wrt các params “theta” của unrolled_model
        vector = [p.grad for p in unrolled_model.parameters()]

        # 4) Tính implicit_grads = hessian_vector_product(...) => gradient hiệu chỉnh do unrolled
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

        # Kết hợp dalpha và implicit_grads
        for g, ig in zip(dalpha, implicit_grads):
            g.sub_(eta * ig)   # g = g - eta*ig

        # Copy gradient vừa tính được về alpha của model “thật”
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                # Tạo grad mới, copy dữ liệu từ g
                v.grad = g.clone().detach()
            else:
                # Ghi đè nội dung grad
                v.grad.copy_(g)

    def _construct_model_from_theta(self, theta):
        """
        Dựng một model_new (cùng cấu trúc với self.model) nhưng 
        params = theta (vector) thay thế cho params gốc.
        """
        model_new = self.model.new()  # Hàm .new() trong model_search.py
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            offset = int(offset)
            v_length = int(v.numel())  # or whatever v_length is

            # Lấy đoạn vector theta tương ứng tham số này
            param = theta[offset : offset + v_length].view(v.size())
            params[k] = param
            offset += v_length

        assert offset == len(theta), "Mismatch in param vector length."

        model_dict.update(params)
        model_new.load_state_dict(model_dict)

        # Thay vì .cuda(), ta dùng to(self._device) 
        model_new.to(self._device)
        return model_new

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        """
        Tính toán Hessian-vector product cho alpha:
          (d/d alpha) [d loss/d theta], 
        xấp xỉ bằng finite-difference: f(theta + eps*v) - f(theta - eps*v)
        """
        R = r / _concat(vector).norm()

        # theta+ = theta + R*v
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.add_(v, alpha=R)

        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

        # theta- = theta+ - 2*R*v = theta - R*v
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.sub_(2. * R * v)

        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

        # Đưa parameters về lại ban đầu (theta)
        with torch.no_grad():
            for p, v in zip(self.model.parameters(), vector):
                p.add_(v, alpha=R)

        # Hessian-vector product ~ (grads_p - grads_n) / (2R)
        return [(gp - gn).div_(2. * R) for gp, gn in zip(grads_p, grads_n)]
