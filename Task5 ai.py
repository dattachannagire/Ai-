import cv2
import face_recognition
import os
import numpy as np

# ---------------- Face Detection and Recognition in a Single Image --------------------

def recognize_faces_in_image(image_path, known_faces_dir):
    """
    Detects and recognizes faces in an image.

    Args:
        image_path (str): Path to the input image.
        known_faces_dir (str): Directory containing images of known people.  Each subdirectory
                              should be named after the person and contain images of them.

    Returns:
        image: The image with bounding boxes and names around recognized faces.
    """

    # 1. Load the image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    # 2. Load known faces and their names
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if not os.path.isdir(person_dir):
            continue  # Skip if it's not a directory

        for image_file in os.listdir(person_dir):
            if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            image_path = os.path.join(person_dir, image_file)
            try:
                person_image = face_recognition.load_image_file(image_path)
                person_encoding = face_recognition.face_encodings(person_image)[0]  # Assumes one face per image
                known_face_encodings.append(person_encoding)
                known_face_names.append(person_name)
            except Exception as e:
                print(f"Error loading or encoding {image_path}: {e}")



    # 3. Compare faces in the image with known faces
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)


    # 4. Draw bounding boxes and names on the image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(rgb_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

    return rgb_image  # Returns the modified image


# Example Usage:
known_faces_directory = "known_faces" # Create this directory.  Put subdirectories named after the people. In each subdirectory, put images of that person.
input_image_path = "test_image.jpg"     # Replace with your image path

# Create the 'known_faces' directory and its subdirectories if they don't exist
if not os.path.exists(known_faces_directory):
    os.makedirs(known_faces_directory)

# Add a simple check for if your test image exists
if not os.path.exists(input_image_path):
    print(f"Error: Image file not found at {input_image_path}")
else:

    modified_image = recognize_faces_in_image(input_image_path, known_faces_directory)

    if modified_image is not None:
        cv2.imshow("Faces Detected", modified_image)
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()
    else:
        print("No faces were detected or an error occurred during processing.")b
