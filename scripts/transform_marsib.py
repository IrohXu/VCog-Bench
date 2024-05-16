import os
import shutil
import cv2
from PIL import Image

input_path = "/home/xucao2/VLM_experiment/VCog/MaRs-IB/task3/tf3"
output_md_path = "/home/xucao2/VLM_experiment/VCog/dataset/task3/tf3/md"
output_pd_path = "/home/xucao2/VLM_experiment/VCog/dataset/task3/tf3/pd"


def check_img_difference(img1, img2, threshold):
    diff = 0
    for i in range(img1.width):
        for j in range(img1.height):
            # Get pixel values
            pixel1 = img1.getpixel((i, j))
            pixel2 = img2.getpixel((i, j))
            # Calculate pixel-wise difference
            diff += sum(abs(c1 - c2) for c1, c2 in zip(pixel1, pixel2))
    # Calculate the average difference per pixel
    num_pixels = img1.width * img1.height
    avg_diff = diff / num_pixels
    
    # Determine if images are similar based on threshold
    if avg_diff < threshold:
        return True
    else:
        return False


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


if os.path.exists(output_md_path):
    shutil.rmtree(output_md_path)
os.mkdir(output_md_path)

if os.path.exists(output_pd_path):
    shutil.rmtree(output_pd_path)
os.mkdir(output_pd_path)  

for img_path in os.listdir(input_path):
    img_str_list = img_path.split(".")[0].split("_")
    
    if len(img_str_list) == 4:
        question_path = img_path
        choice_str_list1 = [img_str_list[0], img_str_list[1], "T1", img_str_list[3], "md"]
        md_choice_str1 = "_".join(choice_str_list1) + ".jpeg"
        
        choice_str_list2 = [img_str_list[0], img_str_list[1], "T2", img_str_list[3], "md"]
        md_choice_str2 = "_".join(choice_str_list2) + ".jpeg"
        
        choice_str_list3 = [img_str_list[0], img_str_list[1], "T3", img_str_list[3], "md"]
        md_choice_str3 = "_".join(choice_str_list3) + ".jpeg"
        
        choice_str_list4 = [img_str_list[0], img_str_list[1], "T4", img_str_list[3], "md"]
        md_choice_str4 = "_".join(choice_str_list4) + ".jpeg"
        
        md_save_path = os.path.join(output_md_path, "_".join(img_str_list))
        os.mkdir(md_save_path)
        os.mkdir(os.path.join(md_save_path, "question"))
        os.mkdir(os.path.join(md_save_path, "choice"))
        os.mkdir(os.path.join(md_save_path, "choiceX"))
        os.mkdir(os.path.join(md_save_path, "answer"))
        os.mkdir(os.path.join(md_save_path, "question", "image"))
        os.mkdir(os.path.join(md_save_path, "choice", "image"))
        os.mkdir(os.path.join(md_save_path, "choiceX", "image"))
        os.mkdir(os.path.join(md_save_path, "answer", "image"))
        
        md_save_path_answer = os.path.join(md_save_path, "answer", "image")
        md_save_path_choice = os.path.join(md_save_path, "choice", "image")
        md_save_path_question = os.path.join(md_save_path, "question", "image")
        md_save_path_choiceX = os.path.join(md_save_path, "choiceX", "image")
        
        shutil.copy(os.path.join(input_path, question_path), os.path.join(md_save_path_question, question_path))
        shutil.copy(os.path.join(input_path, md_choice_str1), os.path.join(md_save_path_choice, md_choice_str1))
        shutil.copy(os.path.join(input_path, md_choice_str2), os.path.join(md_save_path_choice, md_choice_str2))
        shutil.copy(os.path.join(input_path, md_choice_str3), os.path.join(md_save_path_choice, md_choice_str3))
        shutil.copy(os.path.join(input_path, md_choice_str4), os.path.join(md_save_path_choice, md_choice_str4))
        
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, md_choice_str1), os.path.join(md_save_path_choiceX, md_choice_str1))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, md_choice_str2), os.path.join(md_save_path_choiceX, md_choice_str2))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, md_choice_str3), os.path.join(md_save_path_choiceX, md_choice_str3))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, md_choice_str4), os.path.join(md_save_path_choiceX, md_choice_str4))
        
        choice_str_list1 = [img_str_list[0], img_str_list[1], "T1", img_str_list[3], "pd"]
        pd_choice_str1 = "_".join(choice_str_list1) + ".jpeg"
        
        choice_str_list2 = [img_str_list[0], img_str_list[1], "T2", img_str_list[3], "pd"]
        pd_choice_str2 = "_".join(choice_str_list2) + ".jpeg"
        
        choice_str_list3 = [img_str_list[0], img_str_list[1], "T3", img_str_list[3], "pd"]
        pd_choice_str3 = "_".join(choice_str_list3) + ".jpeg"
        
        choice_str_list4 = [img_str_list[0], img_str_list[1], "T4", img_str_list[3], "pd"]
        pd_choice_str4 = "_".join(choice_str_list4) + ".jpeg"
        
        pd_save_path = os.path.join(output_pd_path, "_".join(img_str_list))
        os.mkdir(pd_save_path)
        os.mkdir(os.path.join(pd_save_path, "question"))
        os.mkdir(os.path.join(pd_save_path, "choice"))
        os.mkdir(os.path.join(pd_save_path, "answer"))
        os.mkdir(os.path.join(pd_save_path, "choiceX"))
        os.mkdir(os.path.join(pd_save_path, "question", "image"))
        os.mkdir(os.path.join(pd_save_path, "choice", "image"))
        os.mkdir(os.path.join(pd_save_path, "answer", "image"))
        os.mkdir(os.path.join(pd_save_path, "choiceX", "image"))
        
        pd_save_path_answer = os.path.join(pd_save_path, "answer", "image")
        pd_save_path_choice = os.path.join(pd_save_path, "choice", "image")
        pd_save_path_question = os.path.join(pd_save_path, "question", "image")
        pd_save_path_choiceX = os.path.join(pd_save_path, "choiceX", "image")
        
        shutil.copy(os.path.join(input_path, question_path), os.path.join(pd_save_path_question, question_path))
        shutil.copy(os.path.join(input_path, pd_choice_str1), os.path.join(pd_save_path_choice, pd_choice_str1))
        shutil.copy(os.path.join(input_path, pd_choice_str2), os.path.join(pd_save_path_choice, pd_choice_str2))
        shutil.copy(os.path.join(input_path, pd_choice_str3), os.path.join(pd_save_path_choice, pd_choice_str3))
        shutil.copy(os.path.join(input_path, pd_choice_str4), os.path.join(pd_save_path_choice, pd_choice_str4))  
        
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, pd_choice_str1), os.path.join(pd_save_path_choiceX, pd_choice_str1))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, pd_choice_str2), os.path.join(pd_save_path_choiceX, pd_choice_str2))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, pd_choice_str3), os.path.join(pd_save_path_choiceX, pd_choice_str3))
        replace_corners(os.path.join(input_path, question_path), os.path.join(input_path, pd_choice_str4), os.path.join(pd_save_path_choiceX, pd_choice_str4))
        
        shutil.copy(os.path.join(input_path, pd_choice_str1), os.path.join(pd_save_path_answer, pd_choice_str1)) 
        shutil.copy(os.path.join(input_path, md_choice_str1), os.path.join(md_save_path_answer, md_choice_str1)) 
           
    else:
        pass

        
        
        
        