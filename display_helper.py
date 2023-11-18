import matplotlib.pyplot as plt


def display_image(iteration, display_every, psnrs, rgb_predicted):
    iternums = [i * display_every for i in range(len(psnrs))]
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {iteration}")
    plt.subplot(122)
    plt.plot(iternums, psnrs)
    plt.title("PSNR")
    plt.show()


def save_image(iteration, display_every, psnrs, rgb_predicted):
    iternums = [i for i in range(len(psnrs) * display_every)]
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {iteration}")
    plt.subplot(122)
    plt.plot(iternums, psnrs)
    plt.title("PSNR")
    plt.show()
    plt.savefig('Training_Result.png')

