import os
import random 
import shutil

def split_dataset(train_folder, gt_folder, output_train_folder, output_test_folder, test_size=90):
    # Get a list of all files in the train folder
    all_train_files = os.listdir(train_folder)
    
    # Shuffle the list of files randomly
    random.shuffle(all_train_files)

    # Select the first test_size files for the test set
    test_files = all_train_files[:test_size]

    # Create train and test folders for both images and groundtruths
    os.makedirs(os.path.join(output_train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, 'groundtruth'), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, 'groundtruth'), exist_ok=True)

    # Copy the selected test files and their groundtruths to the test folder
    for test_file in test_files:
        input_train_path = os.path.join(train_folder, test_file)
        input_gt_path = os.path.join(gt_folder, test_file)
        output_test_train_path = os.path.join(output_test_folder, 'images', test_file)
        output_test_gt_path = os.path.join(output_test_folder, 'groundtruth', test_file)

        shutil.copy(input_train_path, output_test_train_path)
        shutil.copy(input_gt_path, output_test_gt_path)

    # Copy the remaining files and their groundtruths to the train folder
    for remaining_file in all_train_files[test_size:]:
        input_train_path = os.path.join(train_folder, remaining_file)
        input_gt_path = os.path.join(gt_folder, remaining_file)
        output_train_train_path = os.path.join(output_train_folder, 'images', remaining_file)
        output_train_gt_path = os.path.join(output_train_folder, 'groundtruth', remaining_file)

        shutil.copy(input_train_path, output_train_train_path)
        shutil.copy(input_gt_path, output_train_gt_path)

if __name__ == '__main__':
    # Set your train folder, groundtruth folder, output train folder, and output test folder
    train_folder = r'C:\Users\Rehman\Desktop\Khalifa\Research\Acquaculture review\Image Enhancement\Datasets\UIEB Dataset\raw-890\raw-890' 
    gt_folder = r'C:\Users\Rehman\Desktop\Khalifa\Research\Acquaculture review\Image Enhancement\Datasets\UIEB Dataset\reference-890\reference-890'
    output_train_folder = r'C:\Users\Rehman\Desktop\Khalifa\Research\Acquaculture review\Image Enhancement\MuLA_GAN-main\MuLA_GAN-main\Dataset\UIEB\train'
    output_test_folder = r'C:\Users\Rehman\Desktop\Khalifa\Research\Acquaculture review\Image Enhancement\MuLA_GAN-main\MuLA_GAN-main\Dataset\UIEB\test'

    # Create the output folders if they don't exist
    os.makedirs(output_train_folder, exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_train_folder, 'groundtruth'), exist_ok=True)
    os.makedirs(output_test_folder, exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_test_folder, 'groundtruth'), exist_ok=True)

    # Specify the number of test images (default is 400)
    test_size = 90

    # Call the split_dataset function
    split_dataset(train_folder, gt_folder, output_train_folder, output_test_folder, test_size)

    print(f"Successfully split the dataset. {test_size} image pairs copied to the test set.")

