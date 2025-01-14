from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="CIFAR-10 Classifier",  # Custom page title
    page_icon="🖼️",  # Favicon
    layout="centered"
)

def main():
    # Page title and introduction
    st.title('CIFAR-10 Web Classifier 🌟')
    st.write(
        """
        Welcome to the **CIFAR-10 Web Classifier**!  
        Upload an image that belongs to one of the **10 classes**:  
        **airplane**, **automobile**, **bird**, **cat**, **deer**, **dog**, **frog**, **horse**, **ship**, or **truck**,  
        and let the model predict its class along with the probabilities.
        """
    )

    # Image upload section
    file = st.file_uploader('Upload an image (JPG or PNG only):', type=['jpg', 'png'])

    if file:
        # Load and display the image
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Resize the image to 32x32 and ensure it has 3 channels (RGB)
        resized_image = image.resize((32, 32))
        img_array = np.array(resized_image) / 255  # Normalize pixel values

        # Ensure the image has 3 channels
        if img_array.ndim == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] != 3:  # Not RGB
            img_array = np.stack([img_array[:, :, i] for i in range(3)], axis=-1)

        # Reshape for model input
        img_array = img_array.reshape((1, 32, 32, 3))

        # Load the pretrained model
        model = tf.keras.models.load_model('cifar10_model.h5')

        # Predict the class
        predictions = model.predict(img_array)

        # CIFAR-10 class names
        cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # Display predictions
        fig, ax = plt.subplots()
        y_pos = np.arange(len(cifar10_classes))
        ax.barh(y_pos, predictions[0], align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cifar10_classes)
        ax.invert_yaxis()
        ax.set_xlabel("Probability")
        ax.set_title('CIFAR-10 Predictions')

        st.pyplot(fig)

        # Detailed project explanation
        st.subheader("📚 Project Explanation")
        st.write("""
        ### **Dataset Overview**  
        The CIFAR-10 dataset consists of **60,000 32x32 color images**, divided into 10 classes:  
        - **Airplane**  
        - **Automobile**  
        - **Bird**  
        - **Cat**  
        - **Deer**  
        - **Dog**  
        - **Frog**  
        - **Horse**  
        - **Ship**  
        - **Truck**  

        ### **Steps in this Web Classifier**  
        1. The user uploads an image.  
        2. The image is resized to 32x32 pixels and normalized.  
        3. The model, pretrained on CIFAR-10, predicts the class of the image.  
        4. The predictions are displayed as probabilities for each class.  

        ### **Note**  
        - The model can only classify images belonging to the above 10 classes.  
        - If the uploaded image does not fit any class, the prediction may be incorrect or uncertain.  
        - Ensure the image quality and orientation are appropriate for best results.  
        """)

        # Handling possible errors
        st.subheader("⚠️ Possible Issues")
        st.write("""
        - **Misclassification:** The model might predict the wrong class if the image is unclear or contains multiple objects.  
        - **Incorrect Format:** Make sure to upload images in **.jpg** or **.png** format.  
        - **Input Size:** The uploaded image is automatically resized to **32x32 pixels**, which may result in some loss of detail.
        """)

    else:
        st.text('You have not uploaded an image yet. Please upload one to see predictions.')

    # Footer
    st.markdown(
        """
        ---
        <div style="text-align: center;">
            <strong>Created by Seema ❤️</strong><br>
            Check out more of my projects on <a href="https://github.com/YourGitHubUsername" target="_blank">GitHub</a>.
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == '__main__':
    main()
