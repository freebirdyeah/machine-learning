import tkinter as tk
from PIL import Image, ImageDraw

# Create a Tkinter window
class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Drawing Canvas")
        
        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg='white')
        self.canvas.pack()
        
        # Initialize drawing state
        self.drawing = False
        self.last_x, self.last_y = None, None
        
        # Event bindings
        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        
        # Button to save the drawing
        save_button = tk.Button(root, text="Save", command=self.save_image)
        save_button.pack()
        
        # Image to save drawing
        self.image = Image.new("L", (280, 280), "white")
        self.draw_instance = ImageDraw.Draw(self.image)
    
    def start_drawing(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.drawing:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="black", width=20)
            
            # Draw on PIL image
            self.draw_instance.line([self.last_x, self.last_y, event.x, event.y], fill="black", width=20)
            
            self.last_x, self.last_y = event.x, event.y
    
    def stop_drawing(self, event):
        self.drawing = False
    
    def save_image(self):
        self.image = self.image.resize((28, 28))  # Resize to MNIST dimensions
        self.image.save("./drawn_image.png")
        print("Image saved as drawn_image.png")


# Run the application
root = tk.Tk()
app = DrawingApp(root)
root.mainloop()

