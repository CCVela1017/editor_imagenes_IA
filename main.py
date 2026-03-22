import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import math
import cv2
import numpy as np

from PIL import Image, ImageTk

class EditorDeImagenesOpenCV:
    def __init__(self, root):
        self.root = root
        self.root.title("Editor de imagenes")
        self.root.geometry("850x750")
        self.root.resizable(False, False)

        self.cv_image_original = None    
        self.cv_image_processed = None   
        self.tk_image = None            
        self.filename = None             

        self.has_rectangle_selection = False 
        self.has_circular_selection = False
        self.blank_image_rectangle = None  
        self.blank_image_circle = None  

        # parte superior
        header_frame = ttk.Frame(root, padding=10)
        header_frame.pack(fill=tk.X, side=tk.TOP)

        title_label = ttk.Label(header_frame, text="Editor de imagenes", font=("Arial", 14, "bold"))
        title_label.pack(pady=5)

        load_area_frame = ttk.Frame(header_frame, padding=5)
        load_area_frame.pack(pady=5)

        self.path_entry = ttk.Entry(load_area_frame, width=30)
        self.path_entry.pack(side=tk.LEFT, padx=(0, 10))

        # carga de imagen
        self.load_button = ttk.Button(load_area_frame, text="Cargar", command=self.cargar_imagen)
        self.load_button.pack(side=tk.LEFT)

        center_frame = ttk.Frame(root, padding=10)
        center_frame.pack(fill=tk.BOTH, expand=True, side=tk.TOP)

        image_container_frame = ttk.Frame(center_frame, borderwidth=2, relief="solid")
        image_container_frame.pack(side=tk.LEFT, padx=10, fill=tk.BOTH, expand=True)

        # cuadrado para la imagen
        self.canvas_width = 400
        self.canvas_height = 350
        self.image_canvas = tk.Canvas(image_container_frame, bg="white", width=self.canvas_width, height=self.canvas_height)
        self.image_canvas.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(center_frame, padding=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        def create_labeled_slider(parent, label_text, row, from_=0, to=255, default=127):
            label = ttk.Label(parent, text=label_text, width=7)
            label.grid(row=row, column=0, sticky=tk.W, pady=2)
            scale = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, length=180, showvalue=0, background="white", troughcolor="white", activebackground="black", command=self.procesar_imagen)
            scale.set(default)
            scale.grid(row=row, column=1, pady=2)
            return scale

        # sliders de RGB
        self.r_scale = create_labeled_slider(control_frame, "R", 0, default=255)
        self.g_scale = create_labeled_slider(control_frame, "G", 1, default=255)
        self.b_scale = create_labeled_slider(control_frame, "B", 2, default=255)
        
        self.blur_scale = create_labeled_slider(control_frame, "Blur", 3, from_=1, to=31, default=1)

        border_label = ttk.Label(control_frame, text="Bordes", width=7)
        border_label.grid(row=4, column=0, sticky=tk.W, pady=(10, 2))

        # boders para sobel con x e y
        def create_border_slider(parent, label_text, row, col):
            label = ttk.Label(parent, text=label_text, width=2)
            label.grid(row=row, column=col, sticky=tk.E, padx=(0, 5))
            scale = tk.Scale(parent, from_=0, to=1, orient=tk.HORIZONTAL, length=120, showvalue=0, background="white", troughcolor="white", activebackground="black", command=self.procesar_imagen)
            scale.grid(row=row, column=col+1, pady=2)
            scale.set(0)
            return scale

        self.border_x_scale = create_border_slider(control_frame, "X", 5, 0)
        self.border_y_scale = create_border_slider(control_frame, "Y", 6, 0)

        bottom_frame = ttk.Frame(root, padding=10)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        bottom_selection_frame = ttk.Frame(bottom_frame, padding=10)
        bottom_selection_frame.pack(side=tk.LEFT, padx=10)

        label_selection = ttk.Label(bottom_selection_frame, text="Seleccion (Visual)", font=("Arial", 10))
        label_selection.grid(row=0, column=0, columnspan=2, sticky=tk.W)

        
        circle_btn = tk.Button(bottom_selection_frame, text="◯", font=("Arial", 12), bg="white", borderwidth=1, relief="solid", width=3, height=1, command=self.toggle_circular_selection)
        circle_btn.grid(row=1, column=0, pady=5, padx=(0, 5))
        square_btn = tk.Button(bottom_selection_frame, text="□", font=("Arial", 12), bg="white", borderwidth=1, relief="solid", width=3, height=1, command=self.toggle_square_selection)
        square_btn.grid(row=1, column=1, pady=5)

        label_color_sel = ttk.Label(bottom_selection_frame, text="Colorear seleccion", font=("Arial", 10))
        label_color_sel.grid(row=0, column=2, columnspan=3, sticky=tk.W, padx=20)

        # rgb de selecciones
        def create_sel_color_slider(parent, label_text, col):
            label = ttk.Label(parent, text=label_text, width=2)
            label.grid(row=1, column=col, sticky=tk.E, padx=(5, 0))
            scale = tk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, length=60, showvalue=0, background="white", troughcolor="white", activebackground="black", command=self.procesar_imagen)
            scale.set(127)
            scale.grid(row=1, column=col+1, pady=5)
            return scale

        self.sel_r_scale = create_sel_color_slider(bottom_selection_frame, "R", 2)
        self.sel_g_scale = create_sel_color_slider(bottom_selection_frame, "G", 3)
        self.sel_b_scale = create_sel_color_slider(bottom_selection_frame, "B", 4)

        def create_sel_coord_slider(parent, label_text, row):
            scale = tk.Scale(parent, from_=0, to=self.canvas_width, orient=tk.HORIZONTAL, length=120, showvalue=0, background="white", troughcolor="white", activebackground="black", command=self.procesar_imagen)
            scale.grid(row=row, column=0, pady=2, padx=(0, 5))
            label = ttk.Label(parent, text=label_text, width=2)
            label.grid(row=row, column=1, sticky=tk.W)
            return scale

        # posicion de la seleccion
        coords_frame = ttk.Frame(bottom_selection_frame)
        label_scale_title = ttk.Label(coords_frame, text="Tamaño de seleccion:", font=("Arial", 10))
        label_scale_title.grid(row=0, column=0, columnspan=2, sticky=tk.W)
        coords_frame.grid(row=2, column=0, columnspan=2, rowspan=2, pady=10)
        self.sel_x_scale = create_sel_coord_slider(coords_frame, "X", 1)
        self.sel_y_scale = create_sel_coord_slider(coords_frame, "Y", 2)

        label_selection_position = ttk.Label(coords_frame, text="Posicion de seleccion", font=("Arial", 10))
        label_selection_position.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        position_frame = ttk.Frame(bottom_selection_frame)
        position_frame.grid(row=4, column=0, columnspan=2, pady=5)
        self.pos_x_scale = create_sel_coord_slider(position_frame, "X", 0)
        self.pos_y_scale = create_sel_coord_slider(position_frame, "Y", 1)


        # slider de angulo
        bottom_angle_frame = ttk.Frame(bottom_frame, padding=10)
        bottom_angle_frame.pack(side=tk.RIGHT, padx=10)

        label_angle_title = ttk.Label(bottom_angle_frame, text="Angulo", font=("Arial", 10))
        label_angle_title.grid(row=0, column=0, sticky=tk.E, pady=(0, 5))

        label_0 = ttk.Label(bottom_angle_frame, text="0", font=("Arial", 8))
        label_0.grid(row=0, column=1, sticky=tk.W)
        
        # procesar angulo
        self.angle_scale = tk.Scale(bottom_angle_frame, from_=0, to=360, orient=tk.HORIZONTAL, length=150, showvalue=0, background="white", troughcolor="white", activebackground="black", command=self.procesar_imagen)
        self.angle_scale.grid(row=0, column=2, pady=(0, 5))

        label_360 = ttk.Label(bottom_angle_frame, text="360", font=("Arial", 8))
        label_360.grid(row=0, column=3, sticky=tk.E)

        canvas_angle_frame = ttk.Frame(bottom_angle_frame, borderwidth=2, relief="solid")
        canvas_angle_frame.grid(row=1, column=0, columnspan=4, pady=5)

        self.angle_canvas = tk.Canvas(canvas_angle_frame, width=150, height=150, bg="white")
        self.angle_canvas.pack()

        self.draw_angle_canvas_base()
        self.actualizar_visualizador_angulo()

    # formas circular y rectangulo toggle para limpiar y que no se traslape
    def toggle_circular_selection(self): 
        self.has_circular_selection = not self.has_circular_selection
        if self.has_circular_selection:
            self.has_rectangle_selection = False 
        else:
            self.blank_image_circle = None  
        self.procesar_imagen() 

    
    def toggle_square_selection(self):
        self.has_rectangle_selection = not self.has_rectangle_selection
        if self.has_rectangle_selection:
            self.has_circular_selection = False
        else:
            self.blank_image_rectangle = None  
        self.procesar_imagen()


    # pillow y opencv

    def cargar_imagen(self):
        self.filename = filedialog.askopenfilename(title="Seleccionar imagen", filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg")])
        if not self.filename:
            return

        self.path_entry.delete(0, tk.END)
        self.path_entry.insert(0, self.filename)

        self.cv_image_original = cv2.imread(self.filename)
        
        if self.cv_image_original is None:
            print("Error: No se pudo cargar la imagen.")
            return

        h, w = self.cv_image_original.shape[:2]
        ratio_w = self.canvas_width / w
        ratio_h = self.canvas_height / h
        ratio = min(ratio_w, ratio_h)
        new_size = (int(w * ratio), int(h * ratio))
        
        self.cv_image_original = cv2.resize(self.cv_image_original, new_size, interpolation=cv2.INTER_AREA)
        
        self.procesar_imagen()

    def procesar_imagen(self, _=None):
        if self.cv_image_original is None:
            return

        img = self.cv_image_original.copy()

        self.actualizar_visualizador_angulo()

        # angulos
        angle = self.angle_scale.get()
        if angle != 0:
            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # rgb general
        r_factor = self.r_scale.get() / 255.0
        g_factor = self.g_scale.get() / 255.0
        b_factor = self.b_scale.get() / 255.0

        b_chan, g_chan, r_chan = cv2.split(img)
        
        r_chan = cv2.convertScaleAbs(r_chan, alpha=r_factor)
        g_chan = cv2.convertScaleAbs(g_chan, alpha=g_factor)
        b_chan = cv2.convertScaleAbs(b_chan, alpha=b_factor)
        
        img = cv2.merge((b_chan, g_chan, r_chan))

        # gaussianblur
        blur_val = self.blur_scale.get()
        if blur_val % 2 == 0:
            blur_val += 1
        if blur_val > 1:
            img = cv2.GaussianBlur(img, (blur_val, blur_val), 0)

        # intensidad del sobelk en x e y
        threshold_x = self.border_x_scale.get()
        threshold_y = self.border_y_scale.get()
       

        abs_sobelx = abs_sobely = None
        result = img.copy()  
        applied_edge = False

        gray_img = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        sobelx = cv2.Sobel(gray_img, cv2.CV_8U, 1, 0, ksize=3)
        abs_sobelx = cv2.convertScaleAbs(sobelx)

        sobely = cv2.Sobel(gray_img, cv2.CV_8U, 0, 1, ksize=3)
        abs_sobely = cv2.convertScaleAbs(sobely)

        # verificar que coordenadas del sobel se aplican

        if [threshold_x, threshold_y].count(0) == 0:
            result = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
            applied_edge = True
        
        else:
            if threshold_x == 0 and threshold_y != 0:
                result = abs_sobely
                applied_edge = True


            elif threshold_y == 0 and threshold_x != 0:
                result = abs_sobelx
                applied_edge = True

            else:
                result = img  



        if applied_edge:
            result = cv2.add(result, 128)  
            img = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)  

        if self.has_rectangle_selection:
            self.blank_image_rectangle = np.zeros(img.shape, dtype=np.uint8)
            x_sca = self.sel_x_scale.get()
            y_sca = self.sel_y_scale.get()
            x_pos = self.pos_x_scale.get()
            y_pos = self.pos_y_scale.get()
            self.blank_image_rectangle[y_pos:y_pos+y_sca, x_pos:x_pos+x_sca, 0] = self.sel_r_scale.get()
            self.blank_image_rectangle[y_pos:y_pos+y_sca, x_pos:x_pos+x_sca, 1] = self.sel_g_scale.get()
            self.blank_image_rectangle[y_pos:y_pos+y_sca, x_pos:x_pos+x_sca, 2] = self.sel_b_scale.get()
            img = cv2.addWeighted(img, 1, self.blank_image_rectangle, 1, 0)

        if self.has_circular_selection:
            self.blank_image_circle = np.zeros(img.shape, dtype=np.uint8)
            
            radio = self.sel_x_scale.get() 
            x_pos = self.pos_x_scale.get()
            y_pos = self.pos_y_scale.get()
            
            color_bgr = (
                self.sel_b_scale.get(), 
                self.sel_g_scale.get(), 
                self.sel_r_scale.get()
            )
            
            cv2.circle(
                self.blank_image_circle, 
                (x_pos, y_pos), 
                radio, 
                color_bgr, 
                thickness=-1
            )
            
            img = cv2.addWeighted(img, 1, self.blank_image_circle, 1, 0)

        self.cv_image_processed = img
        self.mostrar_imagen_en_canvas()

    def mostrar_imagen_en_canvas(self):
        if self.cv_image_processed is None:
            return

        rgb_image = cv2.cvtColor(self.cv_image_processed, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(rgb_image)
        
        self.tk_image = ImageTk.PhotoImage(pil_image)
        
        canvas_h = self.image_canvas.winfo_height() if self.image_canvas.winfo_height() > 1 else self.canvas_height
        canvas_w = self.image_canvas.winfo_width() if self.image_canvas.winfo_width() > 1 else self.canvas_width
        img_h, img_w = self.cv_image_processed.shape[:2]
        
        self.image_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.tk_image, anchor=tk.CENTER)


    def draw_angle_canvas_base(self):
        cx, cy = 75, 75
        radius = 60
        self.angle_canvas.create_oval(cx-2, cy-2, cx+2, cy+2, fill="black", tags="base")
        self.angle_canvas.create_text(cx, cy - radius - 8, text="90", font=("Arial", 8), tags="base")
        self.angle_canvas.create_text(cx + radius + 8, cy, text="0", font=("Arial", 8), tags="base")
        self.angle_canvas.create_text(cx, cy + radius + 8, text="270", font=("Arial", 8), tags="base")
        self.angle_canvas.create_text(cx - radius - 8, cy, text="180", font=("Arial", 8), tags="base")

    def actualizar_visualizador_angulo(self):
        self.angle_canvas.delete("hand")

        angle_degrees = self.angle_scale.get()
        angle_radians = math.radians(angle_degrees)

        cx, cy = 75, 75
        radius = 55

        x_end = cx + radius * math.cos(angle_radians)
        y_end = cy - radius * math.sin(angle_radians) 

        self.angle_canvas.create_line(cx, cy, x_end, y_end, width=3, fill="black", tags="hand")

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorDeImagenesOpenCV(root)
    root.mainloop()
