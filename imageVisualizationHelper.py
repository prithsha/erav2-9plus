
import torch
import matplotlib.pyplot as plt


def convert_image_tensor_to_numpy_hwc_format(image_data):
    # Image format is CWH (Channel, hight and width)
    # Converting tensor to numpy to HWC (Height, Width , Channel) format
    return image_data.numpy().transpose((1, 2, 0))

def show_processed_images(selected_images, image_classes, rows = 4, cols = 5):
    figure = plt.figure(figsize=(10, 10))
    for i in range(1, cols * rows + 1):
        img, label = selected_images[i-1]
        figure.add_subplot(rows, cols, i)        
        plt.title(image_classes[label])
        plt.axis("off")
        plt.imshow(img)
    plt.show()

def show_images_from_tensor_array(images_data, image_classes, rows=4, cols=5):
    processed_images_data = []
    for i in range(cols * rows): 
        image_data, label = images_data[i]
        if(isinstance(image_data, torch.Tensor)):
            image_data = image_data.cpu()
        if(isinstance(label, torch.Tensor)):
            label = label.cpu()
        image_data = convert_image_tensor_to_numpy_hwc_format(image_data)
        processed_images_data.append((image_data, label))

    show_processed_images(processed_images_data,image_classes,rows, cols)


def randomly_show_images_from_tensor_array(images_data, image_classes, rows=4, cols=5):
    processed_images_data = []
    for i in range(1, cols * rows + 1):
        # Generate a random integer within dataset length
        sample_idx = torch.randint(0, len(images_data), size=(1,))  
        image_data, label = images_data[sample_idx.item()]
        if(isinstance(image_data, torch.Tensor)):
            image_data = image_data.cpu()
        if(isinstance(label, torch.Tensor)):
            label = label.cpu()

        image_data = convert_image_tensor_to_numpy_hwc_format(image_data)
        processed_images_data.append((image_data, label))

    show_processed_images(processed_images_data,image_classes,rows, cols)

def show_image(image_data, label):
    plt.figure(figsize=(4,4))
    plt.title(f"{label}")
    plt.imshow(convert_image_tensor_to_numpy_hwc_format(image_data))
        