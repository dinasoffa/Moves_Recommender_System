import os
import random

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


def generate_posters():
    # Define the size of the image
    width = 300
    height = 400

    # Define the font and font size
    font = ImageFont.truetype('arial.ttf', size=20)

    # Create the folder to save the images
    if not os.path.exists('static/posters'):
        os.makedirs('static/posters')

    # Load the dataframe
    df = pd.read_csv('ml-latest-small/movies.csv')

    # Iterate over the rows in the dataframe
    for index, row in df.iterrows():
        # Get the title from the row
        title = row['title']

        # Wrap the title if necessary
        max_width = width - 20
        words = title.split(' ')
        lines = []
        current_line = ''
        for word in words:
            test_line = current_line + word + ' '
            test_width, _ = font.getsize(test_line)
            if test_width > max_width:
                lines.append(current_line.strip())
                current_line = word + ' '
            else:
                current_line = test_line
        lines.append(current_line.strip())
        title = '\n'.join(lines)

        # Get the size of the wrapped title
        text_width, text_height = font.getsize_multiline(title)

        # If the title is too tall, resize the font
        if text_height > height:
            font_size = int((height / text_height) * font.size)
            font = ImageFont.truetype('arial.ttf', size=font_size)
            text_width, text_height = font.getsize_multiline(title)

        # Calculate the position to draw the title
        x = (width - text_width) / 2
        y = (height - text_height) / 2

        # Create a new image with a random background color
        r, g, b = random.randint(0, 100), random.randint(0, 100), random.randint(0, 100)
        image = Image.new('RGB', (width, height), (r, g, b))

        # Draw the title on the image
        draw = ImageDraw.Draw(image)
        draw.multiline_text((x, y), title, font=font, fill='white', align='center')

        # Save the image
        filename = 'poster_{}.png'.format(row['movieId'])
        file_dest = os.path.join('static', 'posters', filename)
        image.save(file_dest)

        # Update the dataframe with the file destination
        df.at[index, 'poster_dest'] = file_dest

    # Save the updated dataframe
    df.to_csv('ml-latest-small/movies_updated.csv', index=False)
    
    df = pd.read_csv('ml-latest-small/movies_updated.csv')
    df['poster_dest'] = df['poster_dest'].str.replace(r'static\\', '', regex=True)
    df['poster_dest'] = df['poster_dest'].str.replace(r'\\', '/', regex=True)
    df.to_csv('ml-latest-small/movies_updated.csv', index=False)

# Call the function to generate the posters
generate_posters()