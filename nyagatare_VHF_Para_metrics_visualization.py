# Visualizations and metrics
import numpy as np
from sklearn.metrics import accuracy_score

# Assuming you have your model `model` and test data `X_test` already defined and trained

# Predict on test data
y_pred = weak_supervision_model.predict(X_test)

# Convert predictions and ground truth masks back to label format
y_pred_labels = np.argmax(y_pred, axis=-1)
y_test_labels = np.argmax(y_test_cat, axis=-1)

# Get unique labels
labels = np.unique(y_test_labels)

# Calculate accuracy for each class
for label in labels:
    # Filter predictions and ground truth for the current label
    y_pred_label = y_pred_labels[y_test_labels == label]
    y_true_label = y_test_labels[y_test_labels == label]
    
    # Calculate accuracy
    acc = accuracy_score(y_true_label, y_pred_label)
    
    # Print accuracy for the current class
    print(f'Accuracy - Class {label}: {acc}')
    
    
# Accuracy Graph for each class
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Define class names
class_names = ['Green-Blue', 'Water', 'Built-up', 'Vegetation', 'Roads', 'Faremland', 'Forest', 'Wetlands', 'Bareland']

# Assuming you have your model `weak_supervision_model` and test data `X_test` already defined and trained

# Predict on test data
y_pred = weak_supervision_model.predict(X_test)

# Convert predictions and ground truth masks back to label format
y_pred_labels = np.argmax(y_pred, axis=-1)
y_test_labels = np.argmax(y_test_cat, axis=-1)

# Get unique labels
labels = np.unique(y_test_labels)

# Initialize a list to store accuracy values for each class
accuracies = []

# Calculate accuracy for each class
for label in labels:
    # Filter predictions and ground truth for the current label
    y_pred_label = y_pred_labels[y_test_labels == label]
    y_true_label = y_test_labels[y_test_labels == label]
    
    # Calculate accuracy
    acc = accuracy_score(y_true_label, y_pred_label)
    
    # Append accuracy to the list
    accuracies.append(acc)

# Plot line graph for accuracy with class names and customized style
plt.figure(figsize=(10, 6))
plt.plot(class_names, accuracies, marker='o', linestyle='-', linewidth=2, markersize=8)  # Increase line size and marker size
plt.title('Accuracy for Each Class', fontsize=16)  # Increase title font size
plt.xlabel('NLA Classification scheme ', fontsize=14)  # Increase x-axis label font size
plt.ylabel('Accuracy', fontsize=14)  # Increase y-axis label font size
plt.xticks(rotation=45, fontsize=12)  # Rotate x-axis labels and increase font size
plt.yticks(fontsize=12)  # Increase y-axis tick font size
plt.grid(True)
plt.gca().set_facecolor('lightcoral')  # Change background color to red

# Save the plot with DPI=1000
# plt.savefig('accuracy_line_plot.png', dpi=1000)
plt.savefig('../SmallV3/Model/c2f/metrics/red_eachclass_accuracy_line_plot.png', dpi=1000)
plt.show()


# Evaluation of receptive fields
import numpy as np
import matplotlib.pyplot as plt

# Define a function to calculate receptive field sizes
def calculate_receptive_fields(input_size, kernel_sizes, strides):
    receptive_fields = [1]  # Start with receptive field of size 1 for the input layer
    current_size = 1
    for kernel_size, stride in zip(kernel_sizes, strides):
        current_size = current_size + (kernel_size - 1) * receptive_fields[-1]  # Calculate receptive field size
        receptive_fields.append(current_size * stride)  # Update receptive field size for next layer
    return receptive_fields

# Define model architecture parameters (kernel sizes and strides for each layer)
kernel_sizes = [3, 3, 3, 3]  # Example kernel sizes
strides = [2, 2, 2, 2]  # Example strides

# Input image size
input_size = 256

# Calculate receptive fields
receptive_fields = calculate_receptive_fields(input_size, kernel_sizes, strides)

# Visualize receptive fields
plt.figure(figsize=(8, 6))
plt.plot(receptive_fields, marker='o')
plt.title('Evolution of Receptive Fields')
plt.xlabel('Layer Depth')
plt.ylabel('Receptive Field Size')
plt.grid(True)
plt.show()

# Predictions with class boundaries
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
# from matplotlib.colors import ListedColormap

# Define colormap for real class colors
# class_colors = ['#000000', '#0000FF', '#808080', '#008000', '#FF0000', '#FFFF00', '#008080', '#00FF00']
# cmap = ListedColormap(class_colors)

def evaluate_model_performance(model, X_test, y_test):
    # Initialize subplots
    plt.figure(figsize=(18, 18))

    for i in range(6):
        # Generate a random test image index
        test_img_number = random.randint(0, len(X_test) - 1)
        test_img = X_test[test_img_number]
        ground_truth = y_test[test_img_number]
        test_img_input = np.expand_dims(test_img, 0)

        # Make prediction on the test image using the model
        test_pred = model.predict(test_img_input)
        test_prediction = np.argmax(test_pred, axis=3)[0,:,:]

        # Plotting the results
        plt.subplot(6, 3, i*3 + 1)
        plt.title('T-Image')
        plt.imshow(test_img[:,:,0])
        plt.axis('off')

        plt.subplot(6, 3, i*3 + 2)
        plt.title('T-Label')
        plt.imshow(ground_truth[:,:,0], cmap=cmap, vmin=0, vmax=len(class_colors)-1)
        plt.axis('off')

        plt.subplot(6, 3, i*3 + 3)
        plt.title('Model Prediction')
        plt.imshow(test_prediction, cmap=cmap, vmin=0, vmax=len(class_colors)-1)
        plt.axis('off')

    # Adjust layout
    plt.tight_layout()

    # Save the plot with DPI=1000
#     plt.savefig('Model/c2f/predictions/labels_predictions.png', dpi=1000)

    # Show the plot
    plt.show()

# Assuming you have your model `weak_supervision_model` and test data `X_test` and `y_test` already defined and trained
evaluate_model_performance(weak_supervision_model, X_test, y_test)

