import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import numpy as np
import pytesseract  
from PIL import Image, ImageTk

class PlacaDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector de Placas de Vehículos")
        self.root.geometry("1000x700")

        self.canvas_width = 800
        self.canvas_height = 500
        self.image_canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg="gray")
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        self.plate_label = ttk.Label(self.root, text="Texto detectado: None", font=("Arial", 18, "bold"))
        self.plate_label.pack(pady=10)

        self.load_button = ttk.Button(self.root, text="Cargar Imagen", command=self.load_image)
        self.load_button.pack(pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Seleccionar Imagen", filetypes=[("Archivos de Imagen", "*.jpg *.jpeg *.png")])
        if file_path:
            image = cv2.imread(file_path)
            self.plate_label.config(text="Procesando...")
            processed_image = self.detect_plates(image)
            self.display_image(processed_image)

    def detect_plates(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        detected_text = "No se detectó texto"

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.018 * peri, True)

            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)

                if 2 < aspect_ratio < 5:
                    plate_roi = gray[y:y+h, x:x+w]
                    
                    _, plate_bin = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    text = pytesseract.image_to_string(plate_bin, config='--psm 7') # 1 linea de texto
                    detected_text = text.strip()

                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(image, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

        self.plate_label.config(text=f"Placa: {detected_text}")
        return image

    def display_image(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(image_rgb)
        
        img_pil.thumbnail((self.canvas_width, self.canvas_height))
        
        self.tk_image = ImageTk.PhotoImage(img_pil)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(self.canvas_width // 2, self.canvas_height // 2, image=self.tk_image)

if __name__ == "__main__":
    root = tk.Tk()
    app = PlacaDetectorApp(root)
    root.mainloop()