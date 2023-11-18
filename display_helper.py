from matplotlib import pyplot as plt


def display_image(psnrs, rgb_predicted):
    iternums = [i for i in range(len(psnrs))]
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {i}")
    plt.subplot(122)
    plt.plot(iternums, psnrs)
    plt.title("PSNR")
    plt.show()