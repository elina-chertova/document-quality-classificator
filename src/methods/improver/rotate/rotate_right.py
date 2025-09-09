"""
Возможные проблемы: некорректные метаданные pdf файла.

Решение: использование встроенной функции для поворота изображений.
Проблема решения: уничтожение текстового слоя

Иначе используем page.set_rotation
Проблема: не всегда корректная интерпретация изображения, поворот срабатывает не в 100% случаев
"""


import os
import shutil
import fitz
import pytesseract
from PIL import Image
from io import BytesIO


class RightAngleRotation:
    def __init__(self, input_folder, output_folder, failed_folder, lang="rus+eng", dpi=300, rotation_mode="physical"):
        """
        rotation_mode:
            - "logical": использует set_rotation (без потерь, но может не работать корректно)
            - "physical": рендерит изображение и вставляет его в новую страницу (всегда сработает, но теряет текстовый слой)
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.failed_folder = failed_folder
        self.lang = lang
        self.dpi = dpi
        self.rotation_mode = rotation_mode.lower()

    def detect_page_rotation(self, page_image):
        try:
            osd = pytesseract.image_to_osd(page_image)
            for line in osd.split("\n"):
                if "Rotate" in line:
                    return int(line.split(":")[-1].strip())
        except Exception as e:
            print(f"[WARNING] OSD detection failed: {e}")
            return None
        return 0

    def _avg_conf(self, img):
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang=self.lang)
            vals = []
            for conf in data["conf"]:
                if isinstance(conf, str) and conf.strip().isdigit():
                    v = int(conf)
                    if v > 0:
                        vals.append(v)
                elif isinstance(conf, (int, float)) and conf > 0:
                    vals.append(int(conf))
            return sum(vals) / len(vals) if vals else 0.0
        except Exception as e:
            print(f"[WARNING] Confidence check failed: {e}")
            return 0.0

    def is_text_upside_down(self, image):
        conf_normal = self._avg_conf(image)
        conf_rot180 = self._avg_conf(image.rotate(180, expand=True))
        return conf_rot180 > conf_normal

    def render_rotated_page_as_image(self, input_pdf, output_pdf, angle):
        try:
            doc = fitz.open(input_pdf)
            page = doc[0]
            mat = fitz.Matrix(2, 2).prerotate(angle)
            pix = page.get_pixmap(matrix=mat)

            new_doc = fitz.open()
            width, height = pix.width, pix.height
            new_page = new_doc.new_page(width=width, height=height)
            new_page.insert_image(new_page.rect, pixmap=pix)

            new_doc.save(output_pdf)
            new_doc.close()
            doc.close()
            print(f"[OK] Saved rotated PDF (image): {output_pdf}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save rotated PDF (image): {e}")
            return False

    def correct_pdf_orientation(self, input_pdf_path, output_pdf_path):
        print(f"[INFO] Processing: {os.path.basename(input_pdf_path)}")

        try:
            doc = fitz.open(input_pdf_path)
            page = doc[0]
            pix = page.get_pixmap(dpi=self.dpi)
            page_image = Image.open(BytesIO(pix.tobytes("png")))
            page_image.info["dpi"] = (self.dpi, self.dpi)
        except Exception as e:
            print(f"[ERROR] Could not render PDF to image: {e}")
            return False

        rotation_angle = self.detect_page_rotation(page_image)
        if rotation_angle is None:
            print("[INFO] OSD unavailable, skipping rotation")
            return False

        print(f"[INFO] OSD detected rotation {rotation_angle}°")

        if rotation_angle in [90, 180, 270]:
            if self.rotation_mode == "physical":
                doc.close()
                return self.render_rotated_page_as_image(input_pdf_path, output_pdf_path, angle=rotation_angle)
            elif self.rotation_mode == "logical":
                try:
                    for page in doc:
                        if rotation_angle == 90:
                            page.set_rotation(270)
                        elif rotation_angle == 180:
                            page.set_rotation(180)
                        elif rotation_angle == 270:
                            page.set_rotation(90)

                    doc.save(output_pdf_path, garbage=0, deflate=False)
                    doc.close()
                    print(f"[OK] Saved rotated PDF (logical): {output_pdf_path}")
                    return True
                except Exception as e:
                    print(f"[ERROR] Failed to apply logical rotation: {e}")
                    doc.close()
                    return False
            else:
                print(f"[ERROR] Unknown rotation_mode: {self.rotation_mode}")
                doc.close()
                return False
        else:
            print("[INFO] No rotation needed, saving original PDF")
            doc.close()
            shutil.copy(input_pdf_path, output_pdf_path)
            return True

    def process_all(self):
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.failed_folder, exist_ok=True)

        for file_name in os.listdir(self.input_folder):
            if not file_name.lower().endswith(".pdf"):
                continue

            input_path = os.path.join(self.input_folder, file_name)
            output_path = os.path.join(self.output_folder, file_name)

            try:
                success = self.correct_pdf_orientation(input_path, output_path)
                if not success:
                    failed_path = os.path.join(self.failed_folder, file_name)
                    shutil.copy(input_path, failed_path)
                    print(f"[FAILED] Copied to failed folder: {failed_path}")
                    if os.path.exists(output_path):
                        try:
                            os.remove(output_path)
                            print(f"[INFO] Removed incomplete output: {output_path}")
                        except Exception as rm_err:
                            print(f"[WARNING] Could not remove incomplete output: {rm_err}")
            except Exception as e:
                print(f"[ERROR] Unhandled error while processing {file_name}: {e}")
                failed_path = os.path.join(self.failed_folder, file_name)
                try:
                    shutil.copy(input_path, failed_path)
                    print(f"[FAILED] Copied to failed folder: {failed_path}")
                except Exception as copy_error:
                    print(f"[ERROR] Could not copy to failed folder: {copy_error}")








# использует логический поворот
# class RightAngleRotation:
#     def __init__(self, input_folder, output_folder, failed_folder, lang="rus+eng", dpi=300):
#         self.input_folder = input_folder
#         self.output_folder = output_folder
#         self.failed_folder = failed_folder
#         self.lang = lang
#         self.dpi = dpi
#
#     def detect_page_rotation(self, page_image):
#         try:
#             osd = pytesseract.image_to_osd(page_image)
#             for line in osd.split("\n"):
#                 if "Rotate" in line:
#                     return int(line.split(":")[-1].strip())
#         except Exception as e:
#             print(f"[WARNING] OSD detection failed: {e}")
#             return None
#         return 0
#
#     def _avg_conf(self, img):
#         try:
#             data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang=self.lang)
#             vals = []
#             for conf in data["conf"]:
#                 if isinstance(conf, str) and conf.strip().isdigit():
#                     v = int(conf)
#                     if v > 0:
#                         vals.append(v)
#                 elif isinstance(conf, (int, float)) and conf > 0:
#                     vals.append(int(conf))
#             return sum(vals) / len(vals) if vals else 0.0
#         except Exception as e:
#             print(f"[WARNING] Confidence check failed: {e}")
#             return 0.0
#
#     def is_text_upside_down(self, image):
#         conf_normal = self._avg_conf(image)
#         conf_rot180 = self._avg_conf(image.rotate(180, expand=True))
#         return conf_rot180 > conf_normal
#
#     def correct_pdf_orientation(self, input_pdf_path, output_pdf_path):
#         doc = fitz.open(input_pdf_path)
#         print(f"[INFO] Processing: {os.path.basename(input_pdf_path)}")
#
#         total_pages = len(doc)
#         osd_fail_count = 0
#
#         for i, page in enumerate(doc):
#             pix = page.get_pixmap(dpi=self.dpi)
#             page_image = Image.open(BytesIO(pix.tobytes("png")))
#             page_image.info["dpi"] = (self.dpi, self.dpi)
#
#             rotation_angle = self.detect_page_rotation(page_image)
#             if rotation_angle is None:
#                 osd_fail_count += 1
#                 print(f"[INFO] Page {i + 1}: OSD unavailable, skipping initial rotation")
#             else:
#                 print(f"[INFO] Page {i + 1}: OSD detected rotation {rotation_angle}°")
#                 if rotation_angle == 90:
#                     page.set_rotation(270)
#                 elif rotation_angle == 180:
#                     page.set_rotation(180)
#                 elif rotation_angle == 270:
#                     page.set_rotation(90)
#
#             pix_after = page.get_pixmap(dpi=self.dpi)
#             page_image_after = Image.open(BytesIO(pix_after.tobytes("png")))
#             page_image_after.info["dpi"] = (self.dpi, self.dpi)
#
#             new_angle = self.detect_page_rotation(page_image_after)
#             if new_angle is None:
#                 print(f"[INFO] Page {i + 1}: recheck OSD unavailable")
#             else:
#                 print(f"[INFO] Page {i + 1}: recheck detected rotation {new_angle}°")
#
#             if (new_angle in [0, 180] or new_angle is None) and self.is_text_upside_down(page_image_after):
#                 print(f"[INFO] Page {i + 1}: applying additional 180° rotation based on confidence check")
#                 page.set_rotation((page.rotation + 180) % 360)
#
#         if total_pages > 0 and osd_fail_count == total_pages:
#             print(f"[FAILED] OSD failed on all {total_pages} page(s); marking document as failed")
#             doc.close()
#             return False
#
#         doc.save(output_pdf_path, garbage=0, deflate=False)
#         doc.close()
#         print(f"[OK] Saved: {output_pdf_path}")
#         return True
#
#     def process_all(self):
#         os.makedirs(self.output_folder, exist_ok=True)
#         os.makedirs(self.failed_folder, exist_ok=True)
#
#         for file_name in os.listdir(self.input_folder):
#             if not file_name.lower().endswith(".pdf"):
#                 continue
#
#             input_path = os.path.join(self.input_folder, file_name)
#             output_path = os.path.join(self.output_folder, file_name)
#
#             try:
#                 success = self.correct_pdf_orientation(input_path, output_path)
#                 if not success:
#                     failed_path = os.path.join(self.failed_folder, file_name)
#                     try:
#                         shutil.copy(input_path, failed_path)
#                         print(f"[FAILED] Copied to failed folder: {failed_path}")
#                         if os.path.exists(output_path):
#                             try:
#                                 os.remove(output_path)
#                                 print(f"[INFO] Removed incomplete output: {output_path}")
#                             except Exception as rm_err:
#                                 print(f"[WARNING] Could not remove incomplete output: {rm_err}")
#                     except Exception as copy_error:
#                         print(f"[ERROR] Could not copy to failed folder: {copy_error}")
#             except Exception as e:
#                 print(f"[ERROR] Unhandled error while processing {file_name}: {e}")
#                 failed_path = os.path.join(self.failed_folder, file_name)
#                 try:
#                     shutil.copy(input_path, failed_path)
#                     print(f"[FAILED] Copied to failed folder: {failed_path}")
#                 except Exception as copy_error:
#                     print(f"[ERROR] Could not copy to failed folder: {copy_error}")

#
# # Физически рендерит изображение с нужным поворотом и вставляет его в новую страницу
# class RightAngleRotation:
#     def __init__(self, input_folder, output_folder, failed_folder, lang="rus+eng", dpi=300):
#         self.input_folder = input_folder
#         self.output_folder = output_folder
#         self.failed_folder = failed_folder
#         self.lang = lang
#         self.dpi = dpi
#
#     def detect_page_rotation(self, page_image):
#         try:
#             osd = pytesseract.image_to_osd(page_image)
#             for line in osd.split("\n"):
#                 if "Rotate" in line:
#                     return int(line.split(":")[-1].strip())
#         except Exception as e:
#             print(f"[WARNING] OSD detection failed: {e}")
#             return None
#         return 0
#
#     def _avg_conf(self, img):
#         try:
#             data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang=self.lang)
#             vals = []
#             for conf in data["conf"]:
#                 if isinstance(conf, str) and conf.strip().isdigit():
#                     v = int(conf)
#                     if v > 0:
#                         vals.append(v)
#                 elif isinstance(conf, (int, float)) and conf > 0:
#                     vals.append(int(conf))
#             return sum(vals) / len(vals) if vals else 0.0
#         except Exception as e:
#             print(f"[WARNING] Confidence check failed: {e}")
#             return 0.0
#
#     def is_text_upside_down(self, image):
#         conf_normal = self._avg_conf(image)
#         conf_rot180 = self._avg_conf(image.rotate(180, expand=True))
#         return conf_rot180 > conf_normal
#
#     def render_rotated_page_as_image(self, input_pdf, output_pdf, angle):
#         try:
#             doc = fitz.open(input_pdf)
#             page = doc[0]
#             mat = fitz.Matrix(2, 2).prerotate(angle)
#             pix = page.get_pixmap(matrix=mat)
#
#             new_doc = fitz.open()
#             width, height = pix.width, pix.height
#             new_page = new_doc.new_page(width=width, height=height)
#             new_page.insert_image(new_page.rect, pixmap=pix)
#
#             new_doc.save(output_pdf)
#             new_doc.close()
#             doc.close()
#             print(f"[OK] Saved rotated PDF: {output_pdf}")
#             return True
#         except Exception as e:
#             print(f"[ERROR] Failed to save rotated PDF: {e}")
#             return False
#
#     def correct_pdf_orientation(self, input_pdf_path, output_pdf_path):
#         try:
#             doc = fitz.open(input_pdf_path)
#             page = doc[0]
#             pix = page.get_pixmap(dpi=self.dpi)
#             page_image = Image.open(BytesIO(pix.tobytes("png")))
#             page_image.info["dpi"] = (self.dpi, self.dpi)
#             doc.close()
#         except Exception as e:
#             print(f"[ERROR] Could not open or render page: {e}")
#             return False
#
#         rotation_angle = self.detect_page_rotation(page_image)
#         if rotation_angle is None:
#             print("[INFO] OSD unavailable, skipping rotation")
#             return False
#
#         print(f"[INFO] OSD detected rotation {rotation_angle}°")
#
#         if rotation_angle in [90, 180, 270]:
#             rotated = self.render_rotated_page_as_image(input_pdf_path, output_pdf_path, angle=rotation_angle)
#             return rotated
#         else:
#             print("[INFO] No rotation needed, saving original PDF")
#             shutil.copy(input_pdf_path, output_pdf_path)
#             return True
#
#     def process_all(self):
#         os.makedirs(self.output_folder, exist_ok=True)
#         os.makedirs(self.failed_folder, exist_ok=True)
#
#         for file_name in os.listdir(self.input_folder):
#             if not file_name.lower().endswith(".pdf"):
#                 continue
#
#             input_path = os.path.join(self.input_folder, file_name)
#             output_path = os.path.join(self.output_folder, file_name)
#
#             try:
#                 success = self.correct_pdf_orientation(input_path, output_path)
#                 if not success:
#                     failed_path = os.path.join(self.failed_folder, file_name)
#                     shutil.copy(input_path, failed_path)
#                     print(f"[FAILED] Copied to failed folder: {failed_path}")
#                     if os.path.exists(output_path):
#                         try:
#                             os.remove(output_path)
#                             print(f"[INFO] Removed incomplete output: {output_path}")
#                         except Exception as rm_err:
#                             print(f"[WARNING] Could not remove incomplete output: {rm_err}")
#             except Exception as e:
#                 print(f"[ERROR] Unhandled error while processing {file_name}: {e}")
#                 failed_path = os.path.join(self.failed_folder, file_name)
#                 try:
#                     shutil.copy(input_path, failed_path)
#                     print(f"[FAILED] Copied to failed folder: {failed_path}")
#                 except Exception as copy_error:
#                     print(f"[ERROR] Could not copy to failed folder: {copy_error}")
#
