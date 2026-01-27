"""Egocentric view renderer.

Renders synthetic first-person view with arm/hand visualization.
Creates an immersive VR-like rendering from third-person camera data.
"""
from __future__ import annotations

from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
import math

from .transformer import EgocentricFrame


class EgocentricRenderer:
    """Render immersive egocentric view with arms, hands, and objects.
    
    Creates a first-person VR-like visualization showing:
    - Arms as 3D cylindrical meshes
    - Hands with detailed finger rendering
    - Objects with depth-based sizing
    - Environment grid for spatial reference
    """
    
    # Arm segment colors (gradient from shoulder to hand)
    ARM_COLORS = {
        "shoulder": (180, 160, 140),  # Darker at shoulder
        "upper_arm": (190, 170, 150),
        "elbow": (195, 175, 155),
        "forearm": (200, 180, 160),
        "wrist": (210, 190, 170),
        "hand": (220, 200, 180),
    }
    
    # Hand landmark groups
    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
    FINGER_COLORS = {
        "thumb": (200, 180, 160),
        "index": (210, 190, 170),
        "middle": (215, 195, 175),
        "ring": (210, 190, 170),
        "pinky": (200, 180, 160),
    }
    
    def __init__(
        self,
        width: int = 960,
        height: int = 720,
        fov_horizontal: float = 90.0,
        background_color: Tuple[int, int, int] = (25, 25, 35),
        show_grid: bool = True,
        show_horizon: bool = True,
    ):
        """Initialize renderer.
        
        Args:
            width: Output image width
            height: Output image height
            fov_horizontal: Horizontal field of view in degrees
            background_color: Background color (BGR)
            show_grid: Whether to show perspective grid
            show_horizon: Whether to show horizon line
        """
        self.width = width
        self.height = height
        self.fov_h = np.radians(fov_horizontal)
        self.fov_v = self.fov_h * height / width
        self.bg_color = background_color
        self.show_grid = show_grid
        self.show_horizon = show_horizon
        
        # Focal lengths for projection
        self.fx = width / (2 * np.tan(self.fov_h / 2))
        self.fy = height / (2 * np.tan(self.fov_v / 2))
        self.cx = width / 2
        self.cy = height / 2
        
        # Hand connections
        self.hand_connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17),
        ]
        
        # Finger base indices for each finger
        self.finger_bases = {
            "thumb": [1, 2, 3, 4],
            "index": [5, 6, 7, 8],
            "middle": [9, 10, 11, 12],
            "ring": [13, 14, 15, 16],
            "pinky": [17, 18, 19, 20],
        }
        
        self.fingertip_indices = [4, 8, 12, 16, 20]
        
        # Object colors by class (will be assigned dynamically)
        self.object_colors: Dict[int, Tuple[int, int, int]] = {}
        self.color_palette = [
            (100, 200, 100),   # Green
            (200, 150, 100),   # Orange
            (100, 150, 200),   # Blue
            (200, 100, 150),   # Pink
            (150, 200, 150),   # Light green
            (200, 200, 100),   # Yellow
            (150, 100, 200),   # Purple
        ]
    
    def project_to_image(
        self,
        points_ego: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project 3D egocentric points to 2D image coordinates.
        
        Points are in normalized space: x,y in [-1, 1], z is depth (0.1 to 2.0).
        We use simple perspective projection where depth affects scale.
        """
        z = points_ego[:, 2]
        valid = z > 0.01
        
        uv = np.zeros((len(points_ego), 2))
        
        if np.any(valid):
            x = points_ego[valid, 0]
            y = points_ego[valid, 1]
            z_valid = z[valid]
            
            # Simple perspective: divide by z for depth effect
            # Scale factor: closer objects (small z) appear larger
            scale = 1.0 / (z_valid + 0.5)
            
            # Map normalized coords to screen
            u = self.cx + x * self.width * 0.5 * scale
            v = self.cy + y * self.height * 0.5 * scale
            
            uv[valid, 0] = u
            uv[valid, 1] = v
        
        return uv, valid
    
    def render(
        self,
        ego_frame: EgocentricFrame,
        object_classes: Optional[Dict[int, str]] = None,
        show_labels: bool = True,
    ) -> np.ndarray:
        """Render immersive egocentric view.
        
        Args:
            ego_frame: Egocentric frame data
            object_classes: Optional dict mapping object_id -> class_name
            show_labels: Whether to show text labels
            
        Returns:
            Rendered image (H, W, 3) BGR
        """
        # Create canvas with gradient background
        canvas = self._create_background()
        
        # Draw environment
        if self.show_grid:
            self._draw_floor_grid(canvas)
        if self.show_horizon:
            self._draw_horizon(canvas)
        
        # Draw objects (behind arms)
        self._draw_objects(canvas, ego_frame, object_classes)
        
        # Draw arms (upper body only)
        if ego_frame.left_arm_ego is not None:
            self._draw_arm_3d(canvas, ego_frame.left_arm_ego, "left")
        if ego_frame.right_arm_ego is not None:
            self._draw_arm_3d(canvas, ego_frame.right_arm_ego, "right")
        
        # Draw hands
        if ego_frame.left_hand_ego is not None:
            self._draw_hand_3d(
                canvas,
                ego_frame.left_hand_ego,
                ego_frame.left_hand_pinching,
                "left"
            )
        if ego_frame.right_hand_ego is not None:
            self._draw_hand_3d(
                canvas,
                ego_frame.right_hand_ego,
                ego_frame.right_hand_pinching,
                "right"
            )
        
        # Draw HUD overlay
        if show_labels:
            self._draw_hud(canvas, ego_frame)
        
        return canvas
    
    def _create_background(self) -> np.ndarray:
        """Create gradient background."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Vertical gradient (darker at top, lighter at bottom)
        for y in range(self.height):
            factor = y / self.height
            color = tuple(int(c * (0.6 + 0.4 * factor)) for c in self.bg_color)
            canvas[y, :] = color
        
        return canvas
    
    def _draw_horizon(self, canvas: np.ndarray):
        """Draw horizon line."""
        horizon_y = self.height // 2 - 50  # Slightly above center
        
        # Gradient horizon
        for i in range(-3, 4):
            alpha = 1.0 - abs(i) / 4.0
            color = tuple(int(80 * alpha) for _ in range(3))
            cv2.line(canvas, (0, horizon_y + i), (self.width, horizon_y + i), color, 1)
    
    def _draw_floor_grid(self, canvas: np.ndarray):
        """Draw perspective floor grid."""
        grid_color = (50, 50, 60)
        
        # Draw grid lines at different depths
        for z in np.linspace(0.4, 1.5, 6):
            y_world = 0.6  # Floor level (below eye level)
            
            # Cross lines (horizontal at this depth)
            left_pt = np.array([[-1.5, y_world, z]])
            right_pt = np.array([[1.5, y_world, z]])
            
            uv_l, v_l = self.project_to_image(left_pt)
            uv_r, v_r = self.project_to_image(right_pt)
            
            if v_l[0] and v_r[0]:
                pl = (int(uv_l[0, 0]), int(uv_l[0, 1]))
                pr = (int(uv_r[0, 0]), int(uv_r[0, 1]))
                alpha = max(0.3, 1.0 - z / 2.0)
                color = tuple(int(c * alpha) for c in grid_color)
                cv2.line(canvas, pl, pr, color, 1)
        
        # Vertical lines going into distance
        for x in np.linspace(-1.2, 1.2, 7):
            near_pt = np.array([[x, 0.6, 0.3]])
            far_pt = np.array([[x * 0.3, 0.6, 1.5]])
            
            uv_n, v_n = self.project_to_image(near_pt)
            uv_f, v_f = self.project_to_image(far_pt)
            
            if v_n[0] and v_f[0]:
                pn = (int(uv_n[0, 0]), int(uv_n[0, 1]))
                pf = (int(uv_f[0, 0]), int(uv_f[0, 1]))
                cv2.line(canvas, pn, pf, grid_color, 1)
    
    def _draw_arm_3d(self, canvas: np.ndarray, arm_ego: np.ndarray, side: str):
        """Draw arm with 3D cylinder-like appearance."""
        uv, valid = self.project_to_image(arm_ego)
        
        # Get depths for sizing
        depths = arm_ego[:, 2]
        
        # Draw upper arm (shoulder to elbow)
        if valid[0] and valid[1]:
            self._draw_limb_segment(
                canvas,
                uv[0], uv[1],
                depths[0], depths[1],
                self.ARM_COLORS["shoulder"],
                self.ARM_COLORS["elbow"],
                base_thickness=25
            )
        
        # Draw forearm (elbow to wrist)
        if valid[1] and valid[2]:
            self._draw_limb_segment(
                canvas,
                uv[1], uv[2],
                depths[1], depths[2],
                self.ARM_COLORS["elbow"],
                self.ARM_COLORS["wrist"],
                base_thickness=20
            )
        
        # Draw joints
        for i, (color_key, size) in enumerate([
            ("shoulder", 12), ("elbow", 10), ("wrist", 8)
        ]):
            if valid[i]:
                pt = (int(uv[i, 0]), int(uv[i, 1]))
                if self._in_bounds(pt):
                    depth_scale = max(0.3, 1.0 - depths[i] / 2.0)
                    radius = int(size * depth_scale)
                    color = self.ARM_COLORS[color_key]
                    cv2.circle(canvas, pt, radius, color, -1)
                    # Highlight
                    cv2.circle(canvas, pt, radius - 2, 
                              tuple(min(255, c + 30) for c in color), -1)
    
    def _draw_limb_segment(
        self,
        canvas: np.ndarray,
        pt1: np.ndarray, pt2: np.ndarray,
        depth1: float, depth2: float,
        color1: Tuple[int, int, int],
        color2: Tuple[int, int, int],
        base_thickness: int = 20,
    ):
        """Draw a limb segment with depth-based thickness and gradient color."""
        p1 = (int(pt1[0]), int(pt1[1]))
        p2 = (int(pt2[0]), int(pt2[1]))
        
        if not (self._in_bounds(p1) or self._in_bounds(p2)):
            return
        
        # Calculate thickness based on depth
        depth_scale1 = max(0.3, 1.0 - depth1 / 2.0)
        depth_scale2 = max(0.3, 1.0 - depth2 / 2.0)
        
        thickness1 = int(base_thickness * depth_scale1)
        thickness2 = int(base_thickness * depth_scale2)
        
        # Draw gradient segments
        num_segments = 8
        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments
            
            # Interpolate points
            x1 = int(pt1[0] * (1 - t1) + pt2[0] * t1)
            y1 = int(pt1[1] * (1 - t1) + pt2[1] * t1)
            x2 = int(pt1[0] * (1 - t2) + pt2[0] * t2)
            y2 = int(pt1[1] * (1 - t2) + pt2[1] * t2)
            
            # Interpolate color
            color = tuple(int(color1[j] * (1 - t1) + color2[j] * t1) for j in range(3))
            
            # Interpolate thickness
            thickness = int(thickness1 * (1 - t1) + thickness2 * t1)
            
            cv2.line(canvas, (x1, y1), (x2, y2), color, thickness)
    
    def _draw_hand_3d(
        self,
        canvas: np.ndarray,
        hand_ego: np.ndarray,
        is_pinching: bool,
        side: str
    ):
        """Draw detailed hand with fingers."""
        uv, valid = self.project_to_image(hand_ego)
        depths = hand_ego[:, 2]
        
        # Draw palm
        palm_indices = [0, 5, 9, 13, 17]
        palm_points = []
        for idx in palm_indices:
            if valid[idx]:
                palm_points.append((int(uv[idx, 0]), int(uv[idx, 1])))
        
        if len(palm_points) >= 3:
            pts = np.array(palm_points, np.int32)
            cv2.fillPoly(canvas, [pts], self.ARM_COLORS["hand"])
        
        # Draw finger connections with gradient
        for finger_name, indices in self.finger_bases.items():
            color = self.FINGER_COLORS[finger_name]
            
            # Draw from wrist/palm to fingertip
            prev_idx = 0 if finger_name == "thumb" else indices[0] - 1
            for i, idx in enumerate(indices):
                if valid[prev_idx] and valid[idx]:
                    depth_avg = (depths[prev_idx] + depths[idx]) / 2
                    depth_scale = max(0.3, 1.0 - depth_avg / 2.0)
                    
                    # Thickness decreases toward fingertip
                    thickness = int((8 - i * 1.5) * depth_scale)
                    
                    pt1 = (int(uv[prev_idx, 0]), int(uv[prev_idx, 1]))
                    pt2 = (int(uv[idx, 0]), int(uv[idx, 1]))
                    
                    if self._in_bounds(pt1) or self._in_bounds(pt2):
                        cv2.line(canvas, pt1, pt2, color, max(2, thickness))
                
                prev_idx = idx
        
        # Draw fingertips with highlights
        for i, idx in enumerate(self.fingertip_indices):
            if valid[idx]:
                pt = (int(uv[idx, 0]), int(uv[idx, 1]))
                if self._in_bounds(pt):
                    depth_scale = max(0.3, 1.0 - depths[idx] / 2.0)
                    radius = int(6 * depth_scale)
                    
                    # Fingertip color
                    color = (0, 255, 255) if is_pinching and idx in [4, 8] else (230, 210, 190)
                    cv2.circle(canvas, pt, radius, color, -1)
                    
                    # Highlight
                    cv2.circle(canvas, pt, radius - 2, 
                              tuple(min(255, c + 40) for c in color), -1)
        
        # Draw pinch indicator
        if is_pinching and valid[4] and valid[8]:
            thumb_pt = (int(uv[4, 0]), int(uv[4, 1]))
            index_pt = (int(uv[8, 0]), int(uv[8, 1]))
            
            if self._in_bounds(thumb_pt) and self._in_bounds(index_pt):
                center = ((thumb_pt[0] + index_pt[0]) // 2,
                         (thumb_pt[1] + index_pt[1]) // 2)
                
                # Glowing pinch indicator
                for r in range(20, 5, -3):
                    alpha = (20 - r) / 15.0
                    color = (0, int(255 * alpha), int(255 * alpha))
                    cv2.circle(canvas, center, r, color, 2)
                
                cv2.line(canvas, thumb_pt, index_pt, (0, 255, 255), 2)
    
    def _draw_objects(
        self,
        canvas: np.ndarray,
        ego_frame: EgocentricFrame,
        object_classes: Optional[Dict[int, str]] = None,
    ):
        """Draw objects in the egocentric view."""
        for obj_id, obj_pos in ego_frame.objects_ego.items():
            obj_3d = obj_pos.reshape(1, 3)
            uv, valid = self.project_to_image(obj_3d)
            
            if not valid[0]:
                continue
            
            pt = (int(uv[0, 0]), int(uv[0, 1]))
            if not self._in_bounds(pt):
                continue
            
            # Get or assign color
            if obj_id not in self.object_colors:
                color_idx = len(self.object_colors) % len(self.color_palette)
                self.object_colors[obj_id] = self.color_palette[color_idx]
            
            color = self.object_colors[obj_id]
            
            # Size based on depth
            depth = obj_pos[2]
            base_size = 60
            size = int(base_size * max(0.3, 1.0 - depth / 2.0))
            
            # Draw 3D-looking box
            self._draw_3d_box(canvas, pt, size, color, depth)
            
            # Label
            if object_classes and obj_id in object_classes:
                label = object_classes[obj_id]
            else:
                label = f"#{obj_id}"
            
            cv2.putText(
                canvas, label,
                (pt[0] - len(label) * 4, pt[1] - size - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )
    
    def _draw_3d_box(
        self,
        canvas: np.ndarray,
        center: Tuple[int, int],
        size: int,
        color: Tuple[int, int, int],
        depth: float,
    ):
        """Draw a 3D-looking box representing an object."""
        x, y = center
        half = size // 2
        offset = size // 4  # 3D offset
        
        # Front face
        front_pts = np.array([
            [x - half, y - half],
            [x + half, y - half],
            [x + half, y + half],
            [x - half, y + half],
        ], np.int32)
        
        # Back face (offset)
        back_pts = front_pts + np.array([offset, -offset])
        
        # Draw back edges
        darker = tuple(max(0, c - 50) for c in color)
        for i in range(4):
            cv2.line(canvas, tuple(front_pts[i]), tuple(back_pts[i]), darker, 1)
        
        # Draw back face
        cv2.polylines(canvas, [back_pts], True, darker, 1)
        
        # Draw front face
        cv2.rectangle(canvas, 
                     (x - half, y - half), (x + half, y + half),
                     color, 2)
        
        # Fill with semi-transparent color
        overlay = canvas.copy()
        cv2.rectangle(overlay, 
                     (x - half, y - half), (x + half, y + half),
                     color, -1)
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    
    def _draw_hud(self, canvas: np.ndarray, ego_frame: EgocentricFrame):
        """Draw heads-up display overlay."""
        # Semi-transparent HUD background
        hud_height = 80
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (250, hud_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
        
        # Frame info
        y = 20
        cv2.putText(canvas, f"Frame: {ego_frame.frame_idx}", 
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Hand states
        y += 20
        for hand, is_pinching, conf in [
            ("L", ego_frame.left_hand_pinching, ego_frame.left_hand_confidence),
            ("R", ego_frame.right_hand_pinching, ego_frame.right_hand_confidence),
        ]:
            if conf > 0:
                status = "PINCH" if is_pinching else "OPEN"
                color = (0, 255, 255) if is_pinching else (150, 150, 150)
                cv2.putText(canvas, f"{hand}: {status}", 
                           (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y += 20
        
        # Timestamp
        cv2.putText(canvas, f"t: {ego_frame.timestamp:.2f}s",
                   (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def _in_bounds(self, pt: Tuple[int, int]) -> bool:
        """Check if point is within image bounds."""
        return 0 <= pt[0] < self.width and 0 <= pt[1] < self.height


def render_egocentric_overlay(
    original_frame: np.ndarray,
    ego_frame: EgocentricFrame,
    renderer: EgocentricRenderer,
    overlay_position: str = "bottom_right",
    overlay_scale: float = 0.3,
) -> np.ndarray:
    """Render egocentric view as overlay on original frame."""
    ego_view = renderer.render(ego_frame)
    
    h, w = original_frame.shape[:2]
    overlay_w = int(w * overlay_scale)
    overlay_h = int(h * overlay_scale)
    ego_small = cv2.resize(ego_view, (overlay_w, overlay_h))
    
    margin = 10
    if overlay_position == "bottom_right":
        x = w - overlay_w - margin
        y = h - overlay_h - margin
    elif overlay_position == "bottom_left":
        x = margin
        y = h - overlay_h - margin
    elif overlay_position == "top_right":
        x = w - overlay_w - margin
        y = margin
    else:
        x = margin
        y = margin
    
    output = original_frame.copy()
    cv2.rectangle(output, (x - 2, y - 2), 
                 (x + overlay_w + 2, y + overlay_h + 2), (100, 100, 100), 2)
    output[y:y + overlay_h, x:x + overlay_w] = ego_small
    cv2.putText(output, "Egocentric View", (x, y - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    return output
