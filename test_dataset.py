import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Canvas, Scrollbar, Frame, Label, BOTH, LEFT, RIGHT, Y, VERTICAL, Button, filedialog

image_size = (128, 128)

def verify_signature_bundle(model_path, test_directory):
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    if not os.path.exists(test_directory):
        print(f"The test directory {test_directory} does not exist.")
        return
    
    test_images = []
    filenames = []
    predicted_labels = []
    confidences = []  
    original_count = 0
    forged_count = 0
    
    for root, dirs, files in os.walk(test_directory):
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(root, filename)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
                    if img is None:
                        print(f"Warning: Could not read image {img_path}.")
                        continue
                    img = cv2.resize(img, image_size)  
                    img = img / 255.0  
                    img = img.reshape(1, image_size[0], image_size[1], 1) 
                    test_images.append(img)
                    filenames.append(filename)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    if not test_images:
        print("No valid images found in the test directory.")
        return
    
    for i, img in enumerate(test_images):
        try:
            prediction = model.predict(img)
            predicted_class = np.argmax(prediction)  
            confidence = np.max(prediction) * 100  
            result = "Genuine" if predicted_class == 0 else "Forged"
            predicted_labels.append((filenames[i], result, confidence)) 
            confidences.append(confidence)
            if predicted_class == 0:
                original_count += 1
            else:
                forged_count += 1
        except Exception as e:
            print(f"Error making prediction for image {filenames[i]}: {e}")
    
    display_results_window(predicted_labels, original_count, forged_count)

def display_results_window(predicted_labels, original_count, forged_count):
    window = Tk()
    window.title("Signature Verification Results")
    window.configure(bg='#f0f0f0')

    pie_chart_frame = Frame(window, width=400, height=400, bg='#f0f0f0')
    pie_chart_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=20, pady=20)

    scrollable_frame = Frame(window, bg='#f0f0f0')
    scrollable_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=20, pady=20)

    canvas = Canvas(scrollable_frame, bg='#f0f0f0')
    scrollbar = Scrollbar(scrollable_frame, orient=VERTICAL, command=canvas.yview)
    inner_frame = Frame(canvas, bg='#f0f0f0')
    
    inner_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    canvas.create_window((0, 0), window=inner_frame, anchor="nw")

    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    canvas.config(yscrollcommand=scrollbar.set)

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ['Original', 'Forged']
    counts = [original_count, forged_count]
    
    explode = (0.05, 0.05)
    ax.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140, colors=['#66b3ff', '#ff9999'], shadow=True,
           explode=explode, wedgeprops={'edgecolor': 'gray', 'linewidth': 1, 'linestyle': 'solid'})
    ax.set_title('Overall Signature Verification Results', fontsize=14, fontweight='bold')

    canvas_plot = FigureCanvasTkAgg(fig, master=pie_chart_frame)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True)

    header = Label(inner_frame, text="Signature Verification Results", font=("Arial", 12, "bold"), bg='#f0f0f0')
    header.pack(anchor='w', pady=(0, 10))

    for filename, result, confidence in predicted_labels:
        color = 'green' if result == 'Genuine' else 'red'
        label_text = f"{filename}: {result} ({confidence:.2f}%)"
        label = Label(inner_frame, text=label_text, fg=color, font=("Arial", 10), bg='#f0f0f0')
        label.pack(anchor='w', padx=10, pady=5)

    window.mainloop()

def select_and_verify_new_dataset():
    selected_directory = filedialog.askdirectory(title="Select Dataset Directory")
    if selected_directory:
        verify_signature_bundle(model_path, selected_directory)

model_path = r"D:\signature_verification\signature_verification_model.keras"
default_test_directory = r"C:\Users\yashm\Downloads\signatures"

main_window = Tk()
main_window.title("Signature Verification")
main_window.configure(bg='#f0f0f0')

control_frame = Frame(main_window, bg='#f0f0f0')
control_frame.pack(side=RIGHT, fill=BOTH, expand=True, padx=20, pady=20)

default_button = Button(control_frame, text="Verify Default Dataset", command=lambda: verify_signature_bundle(model_path, default_test_directory), font=("Arial", 12), bg='#4caf50', fg='white')
default_button.pack(pady=10)

select_button = Button(control_frame, text="Select and Verify New Dataset", command=select_and_verify_new_dataset, font=("Arial", 12), bg='#2196f3', fg='white')
select_button.pack(pady=10)

main_window.mainloop()
