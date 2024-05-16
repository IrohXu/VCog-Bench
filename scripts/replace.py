from PIL import Image

def replace_corners(big_image_path, small_image_path, output_path):
    # Open the big image
    big_image = Image.open(big_image_path)
    # Open the small image
    small_image = Image.open(small_image_path)

    # Convert images to RGB mode if they are in RGBA mode
    if big_image.mode == 'RGBA':
        big_image = big_image.convert('RGB')
    if small_image.mode == 'RGBA':
        small_image = small_image.convert('RGB')

    # Get the dimensions of both images
    big_width, big_height = big_image.size
    small_width, small_height = small_image.size

    # Check if the small image can fit into the bottom-right corner of the big image
    if big_width < small_width or big_height < small_height:
        print("Error: Small image dimensions are larger than big image dimensions.")
        return

    # Coordinates to crop the bottom-right corner of the big image
    left = big_width - small_width
    top = big_height - small_height
    right = big_width
    bottom = big_height

    # Crop the bottom-right corner of the big image
    cropped_corner = big_image.crop((left, top, right, bottom))

    # Paste the small image onto the cropped corner
    big_image.paste(small_image, (left, top))

    # Save the modified big image
    big_image.save(output_path)

    print("Images merged successfully!")

# Example usage
replace_corners("/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md/tf1_1_M_ss3/question/image/tf1_1_M_ss3.jpeg", "/home/xucao2/VLM_experiment/VCog/dataset/task1/tf1/md/tf1_1_M_ss3/choice/image/tf1_1_T1_ss3_md.jpeg", "output_image.jpg")