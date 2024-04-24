import matplotlib.pyplot as plt
import torch

def adjust_learning_rate(optimizer, step, total_steps, peak_lr, end_lr=1e-6):
    warmup_steps = int(total_steps * 0.2)
    decay_steps = int(total_steps * 0.8)

    if step < warmup_steps:
        lr = peak_lr * step / warmup_steps
    elif step < warmup_steps + decay_steps:
        step_into_decay = step - warmup_steps
        lr = peak_lr * (end_lr / peak_lr) ** (step_into_decay / decay_steps)
    else:
        lr = end_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_lr_schedule():
    optimizer = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=0.001)
    steps = 10000
    peak_lr = 1e-4

    lrs = []
    for step in range(steps):
        adjust_learning_rate(optimizer, step, steps, peak_lr)
        lrs.append(optimizer.param_groups[0]['lr'])

    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig('Learning_Rate_Schedule.png')  # Path to save the image

plot_lr_schedule()

