import pygame
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Loading AI Model and image
model = tf.keras.models.load_model('digit_model_1.h5')
uploaded_image_path = 'drawing.jpg'


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(28, 28), color_mode='grayscale')
    img_array = image.img_to_array(img)
    # plt.imshow(img_array, cmap='gray')
    # plt.show()
    img_array /= 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_number(model, img_array):
    prediction = model.predict(img_array)
    predicted_number = np.argmax(prediction[0])
    probability = prediction[0][predicted_number]
    return predicted_number, probability


def fill_buttons():
    pygame.draw.rect(screen, button_color, (button_x_1, button_y_1, button_width, button_height))
    pygame.draw.rect(screen, button_color, (button_x_2, button_y_2, button_width, button_height))
    screen.blit(button_text_1, (12.5, 560))
    screen.blit(button_text_2, (685, 560))


# Starting drawing window
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))


# All the drawing window stuff
pygame.display.set_caption("Draw a Digit")
white = (255, 255, 255)
black = (0, 0, 0)
button_color = (150, 150, 150)
button_width, button_height = 150, 40
button_x_1, button_y_1 = 10, 550
button_x_2, button_y_2 = 630, 550
font = pygame.font.Font(pygame.font.match_font('arial'), 20)

fade_start_time = None
drawing = False
last_pos = None
screen.fill(black)
button_text_1 = font.render("Check AI Guess", True, black)
button_text_2 = font.render("Clear", True, black)
fill_buttons()


while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button_x_1 <= event.pos[0] <= button_x_1 + button_width and button_y_1 <= event.pos[1] <= button_y_1 + button_height:
                subsurface = screen.subsurface((0, 0, width, button_y_1))
                pygame.image.save(subsurface, "drawing.jpg")

                preprocessed_img = preprocess_image(uploaded_image_path)
                predicted_number, probability = predict_number(model, preprocessed_img)
                predicted_string = f"Prediction: {predicted_number}"
                probability = round(probability * 100, 3)
                print(probability)
                probability_string = f"Certainty: {probability}"
                guess = font.render(predicted_string, True, white)
                certainty = font.render(probability_string + '%', True, white)

                # print(f"Prediction: {predicted_number}")
                # print(f"Certainty: {probability}")
                screen.blit(guess, (200, 560))
                screen.blit(certainty, (350, 560))
            elif button_x_2 <= event.pos[0] <= button_x_2 + button_width and button_y_2 <= event.pos[1] <= button_y_2 + button_height:
                screen.fill(black)
                fill_buttons()
            else:
                drawing = True
                last_pos = event.pos
        if event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.line(screen, white, last_pos, event.pos, 30)
            last_pos = event.pos
        if event.type == pygame.MOUSEBUTTONUP:
            drawing = False
    pygame.display.update()
