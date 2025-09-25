''' Image Processor GUI for Jacquard Fabric (vibe-coded)'''
import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                               QColorDialog, QGroupBox, QGridLayout, QFrame,
                               QSpacerItem, QSizePolicy)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QColor, QFont, QImage
import numpy as np
import cv2
from img2jacquard import generate_noise_like, img2jacquard, color_order


class ColorPickerWidget(QWidget):
    """Elegant color selection widget with modern design"""
    colorChanged = Signal(QColor)
    
    def __init__(self, label_text="Color", initial_color=QColor(255, 0, 0)):
        super().__init__()
        self.current_color = initial_color
        self.init_ui(label_text)
    
    def init_ui(self, label_text):
        layout = QVBoxLayout()
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Color label
        self.label = QLabel(label_text)
        self.label.setFont(QFont("Segoe UI", 9, QFont.Weight.Normal))
        self.label.setStyleSheet("color: #333333;")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)
        
        # Color preview button
        self.color_button = QPushButton()
        self.color_button.setFixedSize(60, 60)
        self.color_button.setCursor(Qt.PointingHandCursor)
        self.color_button.clicked.connect(self.pick_color)
        self.update_color_preview()
        layout.addWidget(self.color_button)
        
        # Color value label
        self.color_value = QLabel(self.get_color_hex())
        self.color_value.setFont(QFont("Consolas", 8))
        self.color_value.setStyleSheet("color: #666666; background: #f9f9f9; padding: 2px 4px; border: 1px solid #e0e0e0;")
        self.color_value.setAlignment(Qt.AlignCenter)
        self.color_value.setMinimumHeight(18)
        layout.addWidget(self.color_value)
        
        self.setLayout(layout)
    
    def pick_color(self):
        color = QColorDialog.getColor(self.current_color, self, "Select Color")
        if color.isValid():
            self.current_color = color
            self.update_color_preview()
            self.color_value.setText(self.get_color_hex())
            self.colorChanged.emit(color)
    
    def update_color_preview(self):
        """Update the color preview button with minimal styling"""
        # Create a simple colored pixmap
        pixmap = QPixmap(60, 60)
        pixmap.fill(self.current_color)
        
        self.color_button.setIcon(pixmap)
        self.color_button.setStyleSheet(f"""
            QPushButton {{
                border: 2px solid #cccccc;
                background: {self.current_color.name()};
            }}
            QPushButton:hover {{
                border: 2px solid #999999;
                background: {self.current_color.name()};
            }}
            QPushButton:pressed {{
                border: 2px solid #666666;
                background: {self.current_color.name()};
            }}
        """)
    
    def get_color_hex(self):
        """Get hex representation of current color"""
        return f"#{self.current_color.rgb():06x}"
    
    def get_color(self):
        """Get current color"""
        return self.current_color


class ImageProcessorGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image_path = None
        self.front_image_array = None  # OpenCV array (BGR)
        self.back_image_array = None   # OpenCV array (BGR)
        self.result_array = None       # OpenCV array (RGB)
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("Image Processor")
        self.setGeometry(100, 100, 900, 600)
        
        # Set minimal application styling
        self.setStyleSheet("""
            QMainWindow {
                background: #ffffff;
            }
            QGroupBox {
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 500;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
                background: #ffffff;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px 0 4px;
                background: #ffffff;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setContentsMargins(12, 12, 12, 12)
        self.setCentralWidget(central_widget)
        
        # Main layout with minimal spacing
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Left panel for controls
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        # Right panel for image display
        right_panel = self.create_image_panel()
        main_layout.addWidget(right_panel, 2)
        
        # Initialize color pickers
        self.setup_color_pickers()
    
    def create_control_panel(self):
        """Create the minimal left control panel"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #cccccc;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Image selection group
        image_group = QGroupBox("Image Selection")
        image_layout = QVBoxLayout(image_group)
        image_layout.setSpacing(6)
        
        self.select_image_btn = QPushButton("Select Front Image")
        self.select_image_btn.clicked.connect(self.select_front_image)
        self.select_image_btn.setStyleSheet("""
            QPushButton {
                background: #f5f5f5;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 8px 16px;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
            QPushButton:pressed {
                background: #dddddd;
            }
            QPushButton:disabled {
                background: #f5f5f5;
                color: #999999;
            }
        """)
        image_layout.addWidget(self.select_image_btn)
        
        self.select_second_image_btn = QPushButton("Select Back Image")
        self.select_second_image_btn.clicked.connect(self.select_back_image)
        self.select_second_image_btn.setStyleSheet("""
            QPushButton {
                background: #f5f5f5;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 8px 16px;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
            QPushButton:pressed {
                background: #dddddd;
            }
            QPushButton:disabled {
                background: #f5f5f5;
                color: #999999;
            }
        """)
        image_layout.addWidget(self.select_second_image_btn)
        
        self.image_info_label = QLabel("No images selected")
        self.image_info_label.setWordWrap(True)
        self.image_info_label.setStyleSheet("""
            QLabel {
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 10px;
                padding: 6px;
                background: #f9f9f9;
                border: 1px solid #e0e0e0;
            }
        """)
        image_layout.addWidget(self.image_info_label)
        
        layout.addWidget(image_group)
        
        # Color selection group
        color_group = QGroupBox("Color Palette")
        color_layout = QHBoxLayout(color_group)
        color_layout.setSpacing(8)
        
        # Create color picker widgets in a row
        self.color_pickers = []
        for i, (label, color) in enumerate([
            ("Color 1", QColor(220, 53, 69)),    # Red
            ("Color 2", QColor(40, 167, 69)),    # Green  
            ("Color 3", QColor(0, 123, 255))     # Blue
        ]):
            color_picker = ColorPickerWidget(label, color)
            color_picker.colorChanged.connect(self.on_color_changed)
            self.color_pickers.append(color_picker)
            color_layout.addWidget(color_picker)
        
        layout.addWidget(color_group)
        
        # Processing group
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)
        process_layout.setSpacing(6)
        
        self.process_btn = QPushButton("Process Image")
        self.process_btn.clicked.connect(self.process_image)
        self.process_btn.setEnabled(False)
        self.process_btn.setStyleSheet("""
            QPushButton {
                background: #333333;
                color: white;
                border: 1px solid #333333;
                padding: 8px 16px;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #444444;
            }
            QPushButton:pressed {
                background: #222222;
            }
            QPushButton:disabled {
                background: #f5f5f5;
                color: #999999;
                border: 1px solid #cccccc;
            }
        """)
        process_layout.addWidget(self.process_btn)
        
        self.save_btn = QPushButton("Save Result")
        self.save_btn.clicked.connect(self.save_processed_image)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet("""
            QPushButton {
                background: #f5f5f5;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 8px 16px;
                font-family: 'Segoe UI';
                font-size: 11px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
            QPushButton:pressed {
                background: #dddddd;
            }
            QPushButton:disabled {
                background: #f5f5f5;
                color: #999999;
            }
        """)
        process_layout.addWidget(self.save_btn)
        
        layout.addWidget(process_group)
        
        # Add stretch to push everything to the top
        layout.addStretch()
        
        return panel
    
    def create_image_panel(self):
        """Create the minimal right image display panel"""
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background: #ffffff;
                border: 1px solid #cccccc;
            }
        """)
        layout = QVBoxLayout(panel)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)
        
        # Front image group
        front_image_group = QGroupBox("Front Image")
        front_image_layout = QVBoxLayout(front_image_group)
        front_image_layout.setSpacing(6)
        
        self.front_image_label = QLabel("No image loaded")
        self.front_image_label.setAlignment(Qt.AlignCenter)
        self.front_image_label.setMinimumHeight(150)
        self.front_image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background: #f9f9f9;
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
        """)
        front_image_layout.addWidget(self.front_image_label)
        
        layout.addWidget(front_image_group)
        
        # Back image group
        back_image_group = QGroupBox("Back Image")
        back_image_layout = QVBoxLayout(back_image_group)
        back_image_layout.setSpacing(6)
        
        self.back_image_label = QLabel("No image loaded")
        self.back_image_label.setAlignment(Qt.AlignCenter)
        self.back_image_label.setMinimumHeight(150)
        self.back_image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background: #f9f9f9;
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
        """)
        back_image_layout.addWidget(self.back_image_label)
        
        layout.addWidget(back_image_group)
        
        # Processed image group
        processed_group = QGroupBox("Result")
        processed_layout = QVBoxLayout(processed_group)
        processed_layout.setSpacing(6)
        
        self.processed_image_label = QLabel("Processed image will appear here")
        self.processed_image_label.setAlignment(Qt.AlignCenter)
        self.processed_image_label.setMinimumHeight(150)
        self.processed_image_label.setStyleSheet("""
            QLabel {
                border: 1px solid #cccccc;
                background: #f9f9f9;
                color: #666666;
                font-family: 'Segoe UI';
                font-size: 11px;
            }
        """)
        processed_layout.addWidget(self.processed_image_label)
        
        layout.addWidget(processed_group)
        
        return panel
    
    def setup_color_pickers(self):
        """Setup color picker connections"""
        for picker in self.color_pickers:
            picker.colorChanged.connect(self.on_color_changed)
    
    def select_front_image(self):
        """Open file dialog to select the front image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Front Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            self.current_image_path = file_path
            self.load_front_image(file_path)
    
    def select_back_image(self):
        """Open file dialog to select the back image"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select Back Image", 
            "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.tiff)"
        )
        
        if file_path:
            self.load_back_image(file_path)
    
    def load_front_image(self, file_path):
        """Load and display the front image"""
        try:
            # Load image with OpenCV (BGR format)
            self.front_image_array = cv2.imread(file_path)
            if self.front_image_array is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Convert BGR to RGB for color detection
            front_image_rgb = cv2.cvtColor(self.front_image_array, cv2.COLOR_BGR2RGB)
            
            # Detect colors from the image and update color pickers
            detected_colors = self.detect_colors_from_image(front_image_rgb)
            self.update_color_pickers_with_detected_colors(detected_colors)
            
            # Generate noise for back image if not already loaded
            if self.back_image_array is None:
                self.generate_back_image_noise()
            
            # Update image info
            height, width = self.front_image_array.shape[:2]
            filename = os.path.basename(file_path)
            self.update_image_info()
            
            # Display thumbnail
            self.display_image_thumbnail(self.front_image_array, self.front_image_label)
            
            # Enable process button
            self.process_btn.setEnabled(True)
            
        except Exception as e:
            self.image_info_label.setText(f"Error loading front image: {str(e)}")
            self.process_btn.setEnabled(False)
    
    def load_back_image(self, file_path):
        """Load and display the back image"""
        try:
            # Load image with OpenCV (BGR format)
            self.back_image_array = cv2.imread(file_path)
            if self.back_image_array is None:
                raise ValueError(f"Could not load image: {file_path}")
            
            # Update image info
            self.update_image_info()
            
            # Display thumbnail
            self.display_image_thumbnail(self.back_image_array, self.back_image_label)
            
        except Exception as e:
            self.image_info_label.setText(f"Error loading back image: {str(e)}")
    
    def generate_back_image_noise(self):
        """Generate random noise for the back image based on the front image"""
        if self.front_image_array is not None:
            try:
                # Convert BGR to RGB for noise generation
                front_image_rgb = cv2.cvtColor(self.front_image_array, cv2.COLOR_BGR2RGB)
                
                # Generate noise using the existing function (expects RGB)
                noise_array_rgb = generate_noise_like(front_image_rgb)
                
                # Convert RGB noise back to BGR for storage
                self.back_image_array = cv2.cvtColor(noise_array_rgb, cv2.COLOR_RGB2BGR)
                
                # Display the noise image
                self.display_image_thumbnail(self.back_image_array, self.back_image_label)
                
            except Exception as e:
                print(f"Error generating noise: {str(e)}")
    
    def ensure_images_same_size(self):
        """Ensure both images have the same size for processing (resize back to match front)"""
        if self.front_image_array is not None and self.back_image_array is not None:
            if self.front_image_array.shape != self.back_image_array.shape:
                # Resize back image array to match front image array size
                height, width = self.front_image_array.shape[:2]
                self.back_image_array = cv2.resize(self.back_image_array, (width, height))
                # Update the display
                self.display_image_thumbnail(self.back_image_array, self.back_image_label)
                self.update_image_info()
    
    def detect_colors_from_image(self, image_array):
        """Detect the 3 distinct colors from the image array"""
        img_array = image_array
        
        # Get all unique colors
        unique_colors = np.unique(img_array.reshape(-1, img_array.shape[2]), axis=0)
        
        # If we have exactly 3 colors, return them
        if len(unique_colors) == 3:
            return [tuple(color) for color in unique_colors]
        
        # If we have more than 3 colors, find the 3 most frequent ones
        elif len(unique_colors) > 3:
            # Count frequency of each color
            color_counts = {}
            for pixel in img_array.reshape(-1, img_array.shape[2]):
                pixel_tuple = tuple(pixel)
                color_counts[pixel_tuple] = color_counts.get(pixel_tuple, 0) + 1
            
            # Sort by frequency and take top 3
            sorted_colors = sorted(color_counts.items(), key=lambda x: x[1], reverse=True)
            return [color[0] for color in sorted_colors[:3]]
        
        # If we have fewer than 3 colors, pad with default colors
        else:
            detected_colors = [tuple(color) for color in unique_colors]
            default_colors = [(1, 194, 83), (12, 88, 17), (255, 142, 246)]
            
            # Add default colors that aren't already present
            for default_color in default_colors:
                if len(detected_colors) < 3 and default_color not in detected_colors:
                    detected_colors.append(default_color)
            
            return detected_colors[:3]
    
    def update_color_pickers_with_detected_colors(self, detected_colors):
        """Update the color pickers with the detected colors from the image"""
        for i, color_picker in enumerate(self.color_pickers):
            if i < len(detected_colors):
                # Convert RGB tuple to QColor
                r, g, b = detected_colors[i]
                qcolor = QColor(r, g, b)
                color_picker.current_color = qcolor
                color_picker.update_color_preview()
                color_picker.color_value.setText(color_picker.get_color_hex())
    
    def update_image_info(self):
        """Update the image info label with current status"""
        info_parts = []
        
        if self.front_image_array is not None:
            height, width = self.front_image_array.shape[:2]
            filename = os.path.basename(self.current_image_path) if self.current_image_path else "Front Image"
            info_parts.append(f"Front: {filename} ({width}x{height})")
        
        if self.back_image_array is not None:
            height, width = self.back_image_array.shape[:2]
            info_parts.append(f"Back: {width}x{height}")
        
        if info_parts:
            self.image_info_label.setText("\n".join(info_parts))
        else:
            self.image_info_label.setText("No images selected")
    
    def display_image_thumbnail(self, image_array, label_widget, max_size=(400, 400)):
        """Display image thumbnail with elegant styling using OpenCV array"""
        try:
            # Convert BGR to RGB for Qt
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Check if it's BGR (OpenCV format) or RGB
                # If it's already RGB (from result), use as is
                if image_array.dtype == np.uint8:
                    # Convert BGR to RGB for Qt display
                    display_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                else:
                    display_array = image_array
            else:
                display_array = image_array
            
            # Resize image for thumbnail
            height, width = display_array.shape[:2]
            if width > max_size[0] or height > max_size[1]:
                # Calculate new size maintaining aspect ratio
                scale = min(max_size[0] / width, max_size[1] / height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_array = cv2.resize(display_array, (new_width, new_height))
            
            # Convert numpy array to QImage
            height, width, channel = display_array.shape
            bytes_per_line = 3 * width
            qimage = QImage(display_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            
            # Scale to fit label while maintaining aspect ratio
            label_widget.setPixmap(pixmap.scaled(
                label_widget.size(), 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            ))
            
            # Clear any text and update styling
            label_widget.setText("")
            label_widget.setStyleSheet("""
                QLabel {
                    border: 1px solid #cccccc;
                    background: white;
                }
            """)
            
        except Exception as e:
            label_widget.setText(f"Error displaying image: {str(e)}")
            label_widget.setStyleSheet("""
                QLabel {
                    border: 1px solid #dc3545;
                    background: #f8d7da;
                    color: #721c24;
                    font-family: 'Segoe UI';
                    font-size: 11px;
                }
            """)
    
    
    def on_color_changed(self, color):
        """Handle color picker changes"""
        # This is where you could add real-time preview updates
        pass
    
    def process_image(self):
        """Process images using img2jacquard function"""
        if self.front_image_array is None or self.back_image_array is None:
            return
        
        try:
            # Ensure images are the same size before processing
            self.ensure_images_same_size()
            
            # Use OpenCV arrays directly (convert BGR to RGB for img2jacquard)
            front_img_rgb = cv2.cvtColor(self.front_image_array, cv2.COLOR_BGR2RGB)
            back_img_rgb = cv2.cvtColor(self.back_image_array, cv2.COLOR_BGR2RGB)
            
            # Get the detected colors from the front image
            detected_colors = self.detect_colors_from_image(front_img_rgb)
            
            # Use detected colors as the color_map parameter
            self.result_array = img2jacquard(front_img_rgb, back_img_rgb, tuple(detected_colors))
            
            # Display the processed image (result_array is already RGB)
            self.display_image_thumbnail(self.result_array, self.processed_image_label)
            
            # Update the processed image label
            self.processed_image_label.setText("")
            
            # Enable save button after processing
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            self.processed_image_label.setText(f"Processing error: {str(e)}")
            self.save_btn.setEnabled(False)
    
    def get_selected_colors(self):
        """Get the currently selected colors as RGB tuples"""
        colors = []
        for picker in self.color_pickers:
            color = picker.get_color()
            colors.append((color.red(), color.green(), color.blue()))
        return colors
    
    def get_current_images(self):
        """Get the currently loaded images"""
        return self.front_image_array, self.back_image_array
    
    def save_processed_image(self):
        """Save the processed image to a file"""
        if self.result_array is None:
            return
        
        try:
            # Generate default filename based on original image
            if self.current_image_path:
                base_name = os.path.splitext(os.path.basename(self.current_image_path))[0]
                default_filename = f"{base_name}_processed.bmp"
            else:
                default_filename = "processed_image.bmp"
            
            # Open save dialog
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Processed Image",
                default_filename,
                "BMP Files (*.bmp);;PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)"
            )
            
            if file_path:
                # Ensure the file has the correct extension
                if not file_path.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    file_path += '.bmp'
                
                # Save the image
                cv2.imwrite(file_path, cv2.cvtColor(self.result_array, cv2.COLOR_RGB2BGR))
                
                # Show success message
                self.processed_image_label.setText(f"Image saved to:\n{os.path.basename(file_path)}")
                
        except Exception as e:
            self.processed_image_label.setText(f"Save error: {str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Jacquard Image Processor")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = ImageProcessorGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
