import numpy as np
import cv2
import PySimpleGUI as sg
import os
import sys
import shutil
from subprocess import call
import tempfile

# - This function runs the entire 4-stage restoration process programmatically from the GUI.
# - It is very similar to 'run.py', but designed to run quietly in the background when the user clicks 'Restore'.
# - It uses subprocess commands to execute the AI models for each stage.
def modify(input_folder, output_folder):
    def run_cmd(command):
        try:
            call(command, shell=True)
        except KeyboardInterrupt:
            print("Process interrupted")
            sys.exit(1)

    gpu1 = "-1"

    # resolve absolute paths
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    main_environment = os.getcwd()

    # Stage 1: Overall Quality Improve
    print("Running Stage 1: Overall restoration")
    os.chdir(os.path.join(main_environment, "Global"))
    stage_1_input_dir = input_folder
    stage_1_output_dir = os.path.join(output_folder, "stage_1_restore_output")
    if not os.path.exists(stage_1_output_dir):
        os.makedirs(stage_1_output_dir)

    mask_dir = os.path.join(stage_1_output_dir, "masks")
    new_input = os.path.join(mask_dir, "input")
    new_mask = os.path.join(mask_dir, "mask")
    
    stage_1_command_1 = (
        f'python detection.py --test_path "{stage_1_input_dir}" --output_dir "{mask_dir}" --input_size full_size --GPU {gpu1}'
    )
    stage_1_command_2 = (
        f'python test.py --Scratch_and_Quality_restore --test_input "{new_input}" --test_mask "{new_mask}" --outputs_dir "{stage_1_output_dir}" --gpu_ids {gpu1}'
    )
    run_cmd(stage_1_command_1)
    run_cmd(stage_1_command_2)

    # Solve the case when there is no face in the old photo
    stage_1_results = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_output_dir = os.path.join(output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    if os.path.exists(stage_1_results):
        for x in os.listdir(stage_1_results):
            img_dir = os.path.join(stage_1_results, x)
            shutil.copy(img_dir, stage_4_output_dir)

    print("Finish Stage 1 ...\n")

    # Stage 2: Face Detection
    print("Running Stage 2: Face Detection")
    os.chdir(os.path.join(main_environment, "Face_Detection"))
    stage_2_input_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_2_output_dir = os.path.join(output_folder, "stage_2_detection_output")
    if not os.path.exists(stage_2_output_dir):
        os.makedirs(stage_2_output_dir)
    stage_2_command = (
        f'python detect_all_dlib.py --url "{stage_2_input_dir}" --save_url "{stage_2_output_dir}"'
    )
    run_cmd(stage_2_command)
    print("Finish Stage 2 ...\n")

    # Stage 3: Face Restore
    print("Running Stage 3: Face Enhancement")
    os.chdir(os.path.join(main_environment, "Face_Enhancement"))
    stage_3_input_mask = "./"
    stage_3_input_face = stage_2_output_dir
    stage_3_output_dir = os.path.join(output_folder, "stage_3_face_output")
    if not os.path.exists(stage_3_output_dir):
        os.makedirs(stage_3_output_dir)
    stage_3_command = (
        f'python test_face.py --old_face_folder "{stage_3_input_face}" --old_face_label_folder "{stage_3_input_mask}" --tensorboard_log --name Setting_9_epoch_100 --gpu_ids {gpu1} --load_size 256 --label_nc 18 --no_instance --preprocess_mode resize --batchSize 4 --results_dir "{stage_3_output_dir}" --no_parsing_map'
    )
    run_cmd(stage_3_command)
    print("Finish Stage 3 ...\n")

    # Stage 4: Warp back
    print("Running Stage 4: Blending")
    os.chdir(os.path.join(main_environment, "Face_Detection"))
    stage_4_input_image_dir = os.path.join(stage_1_output_dir, "restored_image")
    stage_4_input_face_dir = os.path.join(stage_3_output_dir, "each_img")
    stage_4_output_dir = os.path.join(output_folder, "final_output")
    if not os.path.exists(stage_4_output_dir):
        os.makedirs(stage_4_output_dir)
    stage_4_command = (
        f'python align_warp_back_multiple_dlib.py --origin_url "{stage_4_input_image_dir}" --replace_url "{stage_4_input_face_dir}" --save_url "{stage_4_output_dir}"'
    )
    run_cmd(stage_4_command)
    print("Finish Stage 4 ...\n")
    os.chdir(main_environment)
    print("All the processing is done. Please check the results.")


# - This resizes large photos so they can fit nicely inside the GUI window without stretching.
# - We check the image dimensions and scale them down proportionally using OpenCV if they exceed the max size.
def resize_image_for_gui(image, max_size=(400, 400)):
    h, w = image.shape[:2]
    scale = min(max_size[0]/w, max_size[1]/h)
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
    return image

# - This block designs the visual layout of our desktop application using PySimpleGUI.
# - It creates buttons, text labels, and two image placeholders (one for the original, one for the restored photo).
sg.theme('LightGrey1')

images_col = [
    [sg.Text('Photo Restoration Tool', font=('Helvetica', 16, 'bold'))],
    [sg.Text('Input file:'), sg.In(enable_events=True, key='-IN FILE-', size=(40, 1)), sg.FileBrowse()],
    [sg.Button('Restore Photo', key='-MPHOTO-', size=(15, 1)), sg.Button('Exit', size=(10, 1))],
    [sg.Text('Input Image', size=(30, 1), justification='center'), sg.Text('Restored Image', size=(30, 1), justification='center')],
    [sg.Image(filename='', key='-IN-', size=(400, 400)), sg.Image(filename='', key='-OUT-', size=(400, 400))]
]

layout = [[sg.Column(images_col, element_justification='c')]]

window = sg.Window('Photo Restoration GUI', layout, grab_anywhere=True)

filename = None

# - This is the main event loop that keeps the GUI window open and listens for user clicks.
# - If the user selects a file, it updates the preview image on the left.
# - If the user clicks 'Restore Photo', it disables the button, runs the AI pipeline on a copy of the image, and shows the result on the right.
while True:
    event, values = window.read()
    if event in (None, 'Exit'):
        break

    elif event == '-MPHOTO-':
        if not filename or not os.path.exists(filename):
            sg.popup_error("Please select a valid input file first!")
            continue
            
        window['-MPHOTO-'].update(disabled=True)
        window.perform_long_operation(lambda: None, '-RESTORE_START-')
        
        # Create temp directories for input and output to avoid path issues
        temp_dir = tempfile.mkdtemp()
        temp_input = os.path.join(temp_dir, "input")
        temp_output = os.path.join(temp_dir, "output")
        os.makedirs(temp_input)
        
        # Copy selected file to temp input directory
        base_name = os.path.basename(filename)
        shutil.copy(filename, os.path.join(temp_input, base_name))
        
        try:
            modify(temp_input, temp_output)
            
            # Find the output image
            final_output_dir = os.path.join(temp_output, "final_output")
            restored_path = os.path.join(final_output_dir, base_name)
            
            if os.path.exists(restored_path):
                image_out = cv2.imread(restored_path)
                image_out_resized = resize_image_for_gui(image_out)
                window['-OUT-'].update(data=cv2.imencode('.png', image_out_resized)[1].tobytes())
                sg.popup_ok("Restoration complete!\nSaved locally to: " + restored_path)
            else:
                sg.popup_error("Restoration failed, could not find output image.")
        except Exception as e:
            sg.popup_error(f"An error occurred: {str(e)}")
        finally:
            window['-MPHOTO-'].update(disabled=False)

    elif event == '-IN FILE-':
        filename = values['-IN FILE-']
        if filename and os.path.exists(filename):
            try:
                image = cv2.imread(filename)
                if image is not None:
                    image_resized = resize_image_for_gui(image)
                    window['-IN-'].update(data=cv2.imencode('.png', image_resized)[1].tobytes())
                    window['-OUT-'].update(data=b'') # clear output
            except Exception as e:
                print(f"Error loading image: {e}")

window.close()