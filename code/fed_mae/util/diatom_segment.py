import cv2
import numpy as np


class DiatomSegmenter:
    def __init__(self, img_size=224, morph_kernel_size=5, center_threshold_ratio=0.1):
        self.img_size = img_size
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
        self.center_threshold_ratio = center_threshold_ratio
        self.clip_limit = 2
        self.adapt_block_size = 101
        self.min_object_area = 5000

    def segment(self, image_input):
        """支持两种输入方式：文件路径或BGR numpy数组"""
        """返回二值化掩码 (0:背景, 255:前景)"""
        if isinstance(image_input, str):
            img = self._load_image(image_input)
        else:
            # 直接使用输入图像（假设已经是预处理后的BGR格式）
            img = cv2.resize(image_input, (self.img_size, self.img_size))
        processed = self._preprocess(img)
        binary = self._adaptive_threshold(processed)
        refined = self._morphological_optimize(binary)
        final_mask = self._extract_main_object(refined)
        return final_mask
    def _load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return cv2.resize(img, (self.img_size, self.img_size))
    def _preprocess(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
        return cv2.GaussianBlur(enhanced, (5, 5), 0)

    def _adaptive_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, self.adapt_block_size, 2)

    def _morphological_optimize(self, binary):
        opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self.morph_kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self.morph_kernel)
        return cv2.medianBlur(closed, 5)

    def _extract_main_object(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._create_fallback_mask()

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < self.min_object_area:
            return np.zeros_like(mask)

        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return self._create_fallback_mask()

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        center_x, center_y = self.img_size // 2, self.img_size // 2
        dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        if dist > self.center_threshold_ratio * self.img_size:
            return self._create_fallback_mask()

        hull = cv2.convexHull(largest_contour)
        final_mask = np.zeros_like(mask)
        cv2.drawContours(final_mask, [hull], -1, 255, thickness=-1)
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        return cv2.dilate(final_mask, dilation_kernel, iterations=1)

    def _create_fallback_mask(self):
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        cv2.rectangle(mask,
                      (int(self.img_size * 0.1), int(self.img_size * 0.1)),
                      (int(self.img_size * 0.9), int(self.img_size * 0.9)),
                      255, -1)
        return mask