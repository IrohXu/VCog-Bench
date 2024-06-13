from PIL import Image
import os

input_dir = ""
output_dir = ""

for input_path in os.listdir(input_dir):
    input_name = input_path.split(".")[0]
    image_path = os.path.join(input_dir, input_path)
    image = Image.open(image_path)

    # Get image dimensions
    width, height = image.size

    # Calculate the width of each sub-image
    sub_width = width // 4

    # Create a directory to save the sub-images
    output_choice_dir = output_dir + "/" + input_name + "/" + "choice" + "/" + "image"
    os.makedirs(output_choice_dir, exist_ok=True)
    output_answer_dir = output_dir + "/" + input_name + "/" + "answer" + "/" + "image"
    os.makedirs(output_answer_dir, exist_ok=True)
    
    # Cut the image into 4 sub-images and save them
    sub_images = []
    for i in range(4):
        left = i * sub_width
        right = (i + 1) * sub_width
        sub_image = image.crop((left, 0, right, height))
        sub_image_path = os.path.join(output_choice_dir, f"sub_image_{i+1}.png")
        sub_image.save(sub_image_path)
        sub_images.append(sub_image_path)
        
        if i == 3:
            answer_image_path = os.path.join(output_answer_dir, f"sub_image_{i+1}.png")
            sub_image.save(answer_image_path)
    
    