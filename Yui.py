from __future__ import annotations
import time
from typing import Iterable, Iterator
import numpy as np
import pygame

# ---------------------
# Utils
# ---------------------

class Matrix2D(np.ndarray):
    """,
    A 3x3 transformation matrix for 2D graphics.
    This class represents a 3x3 matrix used for transformations in 2D space.
    It supports operations like translation, rotation, scaling, and point transformation.
    It is a subclass of numpy.ndarray, allowing for matrix operations.
    
    Methods:
        - __new__: Creates a new 3x3 transformation matrix.
        - transform_point: Applies the transformation to a 2D point (x, y).
        - identity: Returns the identity matrix.
        - translation: Returns a translation matrix.
        - rotation: Returns a rotation matrix given an angle in radians.
        - scaling: Returns a scaling matrix.
        - decompose: Decomposes the matrix into translation, rotation, and scale.
        - translate: Returns a new matrix with translation applied after this one.
        - rotate: Returns a new matrix with rotation applied after this one.
        - scale: Returns a new matrix with scaling applied after this one.
        - invert: Returns the inverse of the transformation matrix.
    """
    
    def __new__(cls, input_array=None) -> 'Matrix2D':
        """
        Creates a new 3x3 transformation matrix.
        The matrix is initialized to the identity matrix if no input_array is provided.
        The input_array should be a 3x3 array-like structure (list, tuple, or numpy array).
        If the input_array is not 3x3, a ValueError is raised.
        The matrix is stored as a numpy array with float type.
        The class is a subclass of numpy.ndarray, allowing for matrix operations.

        Args:
            input_array (np.ndarray, optional): A 3x3 array-like structure to initialize the matrix. If None, initializes to the identity matrix.

        Raises:
            ValueError: If the input_array is not a 3x3 array.

        Returns:
            Matrix2D: An instance of Matrix2D initialized with the provided input_array or the identity matrix.
        """
        if input_array is None:
            input_array = np.identity(3, dtype=float)
        obj = np.asarray(input_array, dtype=float).view(cls)
        if obj.shape != (3, 3):
            raise ValueError("Matrix2D must be a 3x3 array.")
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        # No additional attributes for now

    @classmethod
    def identity(cls):
        """
        Returns the identity matrix for 2D transformations.
        """
        return cls(np.identity(3))

    @classmethod
    def translation(cls, tx:float, ty: float) -> 'Matrix2D':
        """
        Returns a translation matrix that translates points by (tx, ty).

        Args:
            tx (float): x translation
            ty (float): y translation

        Returns:
            Matrix2D: A translation matrix that translates points by (tx, ty).
        """
        return cls([
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1]
        ])

    @classmethod
    def rotation(cls, theta_rad: float) -> 'Matrix2D':
        """
        Returns a rotation matrix for a given angle in radians.

        Args:
            theta_rad (float): Angle in radians to rotate points.

        Returns:
            Matrix2D: A rotation matrix that rotates points by theta_rad radians.
        """
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        return cls([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

    @classmethod
    def scaling(cls, sx:float, sy:float) -> 'Matrix2D':
        """
        Returns a scaling matrix that scales points by (sx, sy).

        Args:
            sx (float): x scaling factor
            sy (float): x scaling factor

        Returns:
            Matrix2D: A scaling matrix that scales points by (sx, sy).
        """
        return cls([
            [sx, 0,  0],
            [0,  sy, 0],
            [0,  0,  1]
        ])

    def __repr__(self):
        return f"Matrix2D(\n{super().__repr__()}\n)"
    
    def decompose(self) -> tuple:
        """
        Decomposes the matrix into translation, rotation, and scale components.

        Returns:
            tuple: A tuple containing:
                - translation (tx, ty): The translation components.
                - theta (float): The rotation angle in radians.
                - scale (sx, sy): The scaling factors in x and y directions.
        """
        a, c, tx = self[0]
        b, d, ty = self[1]

        # Extract translation directly
        translation = (tx, ty)

        # Compute scale
        sx = np.hypot(a, b)
        sy = np.hypot(c, d)

        # Normalize to remove scale from rotation matrix
        if sx != 0: a_n, b_n = a / sx, b / sx
        else:       a_n, b_n = a, b
        if sy != 0: c_n, d_n = c / sy, d / sy
        else:       c_n, d_n = c, d

        # Compute rotation from normalized matrix (assumes uniform scaling + no skew)
        theta = np.arctan2(b_n, a_n)

        return translation, theta, (sx, sy)
    
    def translate(self, tx:float, ty:float) -> 'Matrix2D':
        """
        Returns a new matrix with translation applied *after* this one.

        Args:
            tx (float): x translation
            ty (float): y translation

        Returns:
            Matrix2D: A new transformation matrix with translation applied after this one.
        """
        return Matrix2D.translation(tx, ty) @ self

    def rotate(self, theta_rad:float) -> 'Matrix2D':
        """
        Returns a new matrix with rotation applied *after* this one.

        Args:
            theta_rad (float): Angle in radians to rotate points.

        Returns:
            Matrix2D: A new transformation matrix with rotation applied after this one.
        """
        return Matrix2D.rotation(theta_rad) @ self

    def scale(self, sx:float, sy:float) -> 'Matrix2D':
        """
        Returns a new matrix with scaling applied *after* this one.

        Args:
            sx (float): x scaling factor
            sy (float): y scaling factor

        Returns:
            Matrix2D: A new transformation matrix with scaling applied after this one.
        """
        return Matrix2D.scaling(sx, sy) @ self

    def invert(self) -> 'Matrix2D':
        """
        Returns the inverse of the transformation matrix.

        Raises:
            ValueError: If the matrix is not invertible.

        Returns:
            Matrix2D: The inverse of the transformation matrix.
        """
        try:
            inv = np.linalg.inv(self)
            return Matrix2D(inv)
        except np.linalg.LinAlgError:
            raise ValueError("Matrix is not invertible.")

class Vector2D(np.ndarray):
    """
    A 2D vector represented as a 3x1 column vector.
    This class represents a 2D vector in homogeneous coordinates, allowing for
    transformations using a 3x3 Matrix2D. It supports basic vector operations
    such as addition, subtraction, scaling, and transformation by a Matrix2D.
    It is a subclass of numpy.ndarray, allowing for array-like operations.
    
    Attributes:
        x (float): The x component of the vector.
        y (float): The y component of the vector.
    Methods:
        - __new__: Creates a new Vector2D instance.
        - transform: Applies a Matrix2D transformation and returns a new Vector2D.
        - magnitude: Returns the distance from another vector or the origin.
        - heading: Returns the angle (in radians) from the origin or another vector.
        - __add__, __sub__, __mul__, __truediv__, __neg__: Basic vector operations.
        - swizzle: Returns a new Vector2D based on a pattern string.
    """
    def __new__(cls, x=0.0, y=0.0) -> 'Vector2D':
        data = np.array([[x], [y], [1.0]], dtype=float)
        obj = np.asarray(data).view(cls)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None: return

    @property
    def x(self) -> float:
        """The x component of the vector."""
        return self[0, 0]

    @x.setter
    def x(self, value) -> None:
        """Sets the x component of the vector."""
        self[0, 0] = value

    @property
    def y(self) -> float:
        """The y component of the vector."""
        return self[1, 0]

    @y.setter
    def y(self, value) -> None:
        """Sets the y component of the vector."""
        self[1, 0] = value

    def transform(self, matrix: Matrix2D) -> 'Vector2D':
        """
        Applies a Matrix2D transformation to this vector and returns a new Vector2D.

        Args:
            matrix (Matrix2D): The transformation matrix to apply.

        Returns:
            Vector2D: A new Vector2D that is the result of applying the transformation.
        """
        result = matrix @ self
        return Vector2D(result[0, 0], result[1, 0])

    def magnitude(self, origin:'Vector2D'=None) -> float:
        """
        Returns the distance from this vector to another vector or the origin.

        Args:
            origin (Vector2D, optional): Another vector to measure distance from. If None, measures from the origin (0, 0).

        Returns:
            float: The distance from this vector to the origin or another vector.
        """
        dx, dy = self.x, self.y
        if origin is not None:
            dx -= origin.x
            dy -= origin.y
        return np.hypot(dx, dy)

    def heading(self, origin:'Vector2D'=None) -> float:
        """
        Returns the angle (in radians) from this vector to another vector or the origin.

        Args:
            origin (Vector2D, optional): Another vector to measure angle from. If None, measures from the origin (0, 0).

        Returns:
            float: The angle in radians from this vector to the origin or another vector.
        """
        dx, dy = self.x, self.y
        if origin is not None:
            dx -= origin.x
            dy -= origin.y
        return np.arctan2(dy, dx)

    def __add__(self, other:'Vector2D') -> 'Vector2D':
        """
        Adds another Vector2D to this vector and returns a new Vector2D.

        Args:
            other (Vector2D): Another vector to add.

        Returns:
            Vector2D: A new Vector2D that is the sum of this vector and the other vector.
        """
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other:'Vector2D') -> 'Vector2D':
        """
        Subtracts another Vector2D from this vector and returns a new Vector2D.

        Args:
            other (Vector2D): Another vector to subtract.

        Returns:
            Vector2D: A new Vector2D that is the difference of this vector and the other vector.
        """
        return Vector2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Multiplies this vector by a scalar or another Vector2D and returns a new Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value or another Vector2D to multiply with.

        Returns:
            Vector2D: A new Vector2D that is the product of this vector and the scalar or vector.
        """
        return Vector2D(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Allows scalar multiplication with Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value or another Vector2D to multiply with.

        Returns:
            Vector2D: A new Vector2D that is the product of the scalar and this vector.
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar:'Vector2D'|float) -> 'Vector2D':
        """
        Divides this vector by a scalar and returns a new Vector2D.

        Args:
            scalar (Vector2D | float): A scalar value to divide by.

        Returns:
            Vector2D: A new Vector2D that is the result of dividing this vector by the scalar.
        """
        return Vector2D(self.x / scalar, self.y / scalar)

    def __neg__(self) -> 'Vector2D':
        """
        Negates this vector and returns a new Vector2D.

        Returns:
            Vector2D: A new Vector2D that is the negation of this vector.
        """
        return Vector2D(-self.x, -self.y)

    def __repr__(self) -> str:
        """
        Returns a string representation of the Vector2D.

        Returns:
            str: A string representation of the Vector2D in the format "Vector2D(x=..., y=...)".
        """
        return f"Vector2D(x={self.x}, y={self.y})"
        
    def __rmatmul__(self, matrix:Matrix2D) -> 'Vector2D':
        """
        Allows matrix multiplication with a Matrix2D on the left side.

        Args:
            matrix (Matrix2D): The transformation matrix to apply.

        Returns:
            Vector2D: A new Vector2D that is the result of applying the transformation matrix to this vector.
            
        Raises:
            NotImplemented: If the left operand is not a Matrix2D.
        """
        if isinstance(matrix, Matrix2D):
            result = matrix @ self
            return Vector2D(result[0, 0], result[1, 0])
        return NotImplemented
    
    def swizzle(self, pattern:str) -> 'Vector2D':
        """
        Returns a new Vector2D based on a swizzle pattern string.
        The pattern can contain:
            - 'x' for x component
            - 'y' for y component
            - '0' for zero
            - '1' for one
            - 'n1' for negative one
            - 'nx' for negative x component
            - 'ny' for negative y component
        The pattern must be exactly 2 characters long, and each character must be one of the above tokens.
        
        Args:
            pattern (str): A 2-character string representing the swizzle pattern.

        Raises:
            ValueError: If the pattern is not exactly 2 characters long or contains invalid tokens.

        Returns:
            Vector2D: A new Vector2D created based on the swizzle pattern.
        """
        if len(pattern) != 2:
            raise ValueError("Swizzle pattern must be length 2.")

        def get_val(token):
            if token == 'x':
                return self.x
            elif token == 'y':
                return self.y
            elif token == '0':
                return 0.0
            elif token == '1':
                return 1.0
            elif token == 'n1':
                return -1.0
            elif token == 'nx':
                return -self.x
            elif token == 'ny':
                return -self.y
            else:
                raise ValueError(f"Invalid swizzle token: {token}")

        # We need to parse tokens: either 1 or 2 chars, so:
        # 'n1', 'nx', 'ny' are 2-char tokens
        # 'x', 'y', '0', '1' are 1-char tokens

        # Parse the pattern into tokens accordingly:
        tokens = []
        i = 0
        while i < len(pattern):
            # Check for 2-char tokens starting with 'n'
            if pattern[i] == 'n' and i + 1 < len(pattern):
                tokens.append(pattern[i:i+2])
                i += 2
            else:
                tokens.append(pattern[i])
                i += 1

        if len(tokens) != 2:
            raise ValueError("Swizzle pattern must resolve to exactly 2 tokens.")

        x_val = get_val(tokens[0])
        y_val = get_val(tokens[1])
        return Vector2D(x_val, y_val)

    @classmethod
    def random(cls, low:float=0.0, high:float=1.0) -> 'Vector2D':
        """
        Returns a new Vector2D with random x and y components within the specified range.

        Args:
            low (float): The lower bound for the random values (inclusive).
            high (float): The upper bound for the random values (exclusive).

        Returns:
            Vector2D: A new Vector2D with random x and y components.
        """
        return cls(np.random.uniform(low, high), np.random.uniform(low, high))
    
    @classmethod
    def random_unit(cls) -> 'Vector2D':
        """
        Returns a new Vector2D with random x and y components that form a unit vector.

        Returns:
            Vector2D: A new Vector2D with random direction but unit length.
        """
        angle = np.random.uniform(0, 2 * np.pi)
        return cls(np.cos(angle), np.sin(angle))
    
    @classmethod
    def polar(cls, radius:float, angle_rad:float) -> 'Vector2D':
        """
        Returns a new Vector2D from polar coordinates.

        Args:
            radius (float): The distance from the origin.
            angle_rad (float): The angle in radians.

        Returns:
            Vector2D: A new Vector2D representing the point in Cartesian coordinates.
        """
        return cls(radius * np.cos(angle_rad), radius * np.sin(angle_rad))

class Color(pygame.Color):
    def __new__(cls, r=0, g=0, b=0, a=255):
        """
        Creates a new Color instance with the specified RGB(A) values.
        
        Args:
            r (int): Red component (0-255).
            g (int): Green component (0-255).
            b (int): Blue component (0-255).
            a (int, optional): Alpha component (0-255). Defaults to 255 (opaque).
        
        Returns:
            Color: A new Color instance.
        """
        return super().__new__(cls, r, g, b, a)
    
    def __repr__(self):
        """
        Returns a string representation of the Color instance.
        
        Returns:
            str: A string representation of the Color in the format "Color(r, g, b, a)".
        """
        return f"Color({self.r}, {self.g}, {self.b}, {self.a})"
    
    def __str__(self):
        return super().__str__()
    
    def to_tuple(self) -> tuple:
        """
        Returns the color as a tuple (r, g, b, a).
        
        Returns:
            tuple: A tuple containing the RGBA components of the color.
        """
        return (self.r, self.g, self.b, self.a)
    
    def to_hex(self) -> str:
        """
        Returns the color as a hexadecimal string.
        
        Returns:
            str: A hexadecimal string representation of the color, e.g., "#RRGGBBAA".
        """
        return f"#{self.r:02X}{self.g:02X}{self.b:02X}{self.a:02X}"
    
    @classmethod
    def from_hex(cls, hex_str: str) -> 'Color':
        """
        Creates a Color instance from a hexadecimal string.
        
        Args:
            hex_str (str): A hexadecimal string in the format "#RRGGBB" or "#RRGGBBAA".
        
        Returns:
            Color: A new Color instance created from the hexadecimal string.
        
        Raises:
            ValueError: If the hex_str is not in a valid format.
        """
        if hex_str.startswith('#'):
            hex_str = hex_str[1:]
        if len(hex_str) == 6:
            return cls(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16))
        elif len(hex_str) == 8:
            return cls(int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16), int(hex_str[6:8], 16))
        else:
            raise ValueError("Hex string must be in format '#RRGGBB' or '#RRGGBBAA'.")

# ---------------------
# Graphics
# ---------------------

# Global flag to determine if OpenGL should be used
_use_gl = False  # Default to not using OpenGL
def use_gl(use:bool|None=None) -> None|bool:
    """
    Sets or gets the OpenGL usage flag for the graphics module.
    
    Args:
        use (bool, optional): If provided, sets the OpenGL usage flag. If None, returns the current flag.
    
    Returns:
        bool: The current OpenGL usage flag if no argument is provided.
    """
    global _use_gl
    if use is not None:
        _use_gl = use
    return _use_gl

# Also allows pixel access like a numpy array
class Graphics(pygame.surface.Surface):
    """
    A graphics surface that can be used for drawing shapes, images, and text.
    This class is a subclass of pygame.Surface and provides additional methods
    for drawing common shapes and handling transformations.
    
    Variables:
        width (int): The width of the surface.
        height (int): The height of the surface.
        
    """
    
    MODES = ["corner", "corners", "center", "radius"]
    
    def __new__(cls, width:int, height:int) -> 'Graphics':
        """
        Creates a new Graphics surface with the specified width and height.
        
        Args:
            width (int): The width of the surface.
            height (int): The height of the surface.
            flags (int, optional): Pygame surface flags. Defaults to 0.
            depth (int, optional): The bit depth of the surface. Defaults to 0.
        
        Returns:
            Graphics: A new Graphics instance.
        """
        
        if _use_gl:
            # If using OpenGL, create a surface with OpenGL flags
            flags = pygame.OPENGL | pygame.DOUBLEBUF
        else:
            # Create a regular pygame surface
            flags = 0
        surface = pygame.display.set_mode((width, height), flags=flags, depth=0)
        obj = surface.view(cls)
        
        # Add additional attributes
        obj._transfroms = [Matrix2D.identity()]
        
        obj._fill_color = Color(0, 0, 0, 255)  # Default fill color
        obj._stroke_color = Color(255, 255, 255, 255)
        obj._stroke_width = 1  # Default stroke width
        obj._texture = None  # Default texture is None, can be a Graphics object
        obj._gradient = None  # Default gradient is None, can be a tuple of two colors
        
        obj._rect_mode = 'corner'  # Default rectangle mode
        obj._ellipse_mode = 'center'  # Default ellipse mode
        obj._image_mode = 'corner'  # Default image mode
        
        obj._text_align_x = 0  # Horizontal text alignment (0: left, 1: right)
        obj._text_align_y = 0  # Vertical text alignment (0: top, 1: bottom)
        obj._text_font = pygame.font.Font(None, 36)  # Default font
        obj._text_size = 12  # Default text size
        obj._text_leading = 0  # Default text leading (line spacing)
        
        obj._curve_detail = 100  # Default curve detail for bezier curves
        
        return obj
    
    def __array_finalize__(self, obj) -> None:
        if obj is None: return
        # No additional attributes for now
    
    def __repr__(self) -> str:
        """
        Returns a string representation of the Graphics surface.
        
        Returns:
            str: A string representation of the Graphics surface in the format "Graphics(width, height)".
        """
        return f"Graphics({self.get_width()}, {self.get_height()})"
    
    
    # Properties
    @property
    def width(self) -> int:
        """
        Returns the width of the Graphics surface.
        
        Returns:
            int: The width of the surface.
        """
        return self.get_width()
    
    @property
    def height(self) -> int:
        """
        Returns the height of the Graphics surface.
        
        Returns:
            int: The height of the surface.
        """
        return self.get_height()
    
    @property
    def last_transform(self) -> Matrix2D:
        """
        Returns the last transformation matrix applied to the Graphics surface.
        
        Returns:
            Matrix2D: The last transformation matrix.
        """
        return self._transfroms[-1]
    
    
    @staticmethod
    def _coordinates(mode:str, x1:float, y1:float, x2:float, y2:float) -> tuple:
        """
        Converts coordinates based on the specified mode.
        The modes can be:
            - 'corner': (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            - 'corners': (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            - 'center': (x1, y1) is the center, (x2, y2) is the size
            - 'radius': (x1, y1) is the center, (x2, y2) is the radius

        Args:
            mode (str): The mode for coordinate conversion. Can be 'corner', 'corners', 'center', or 'radius'.
            x1 (float)
            y1 (float)
            x2 (float)
            y2 (float)

        Raises:
            ValueError: If the mode is not one of the expected values.

        Returns:
            tuple: A tuple representing the coordinates in the format (x, y, width, height) based on the mode.
        """
        if mode == 'corner':
            # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            return (x1, y1, x2, y2)
        elif mode == 'corners':
            # (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner
            return (x1, y1, x2 - x1, y2 - y1)
        elif mode == 'center':
            # (x1, y1) is the center, (x2, y2) is the size
            return (x1 - x2 / 2, y1 - y2 / 2, x2, y2)
        elif mode == 'radius':
            # (x1, y1) is the center, (x2, y2) is the radius
            return (x1 - x2, y1 - y2, x2 * 2, y2 * 2)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
    
    @property
    def rect_mode(self) -> str:
        """
        Returns the current rectangle mode.
        
        Returns:
            str: The current rectangle mode ('corner', 'corners', 'center', or 'radius').
        """
        return self._rect_mode
    
    @rect_mode.setter
    def rect_mode(self, mode:str) -> None:
        """
        Sets the rectangle mode for drawing rectangles.
        
        Args:
            mode (str): The rectangle mode to set. Can be 'corner', 'corners', 'center', or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['corner', 'corners', 'center', 'radius']:
            raise ValueError(f"Invalid rectangle mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
        self._rect_mode = mode
    
    @property
    def ellipse_mode(self) -> str:
        """
        Returns the current ellipse mode.
        
        Returns:
            str: The current ellipse mode ('center' or 'radius').
        """
        return self._ellipse_mode
    
    @ellipse_mode.setter
    def ellipse_mode(self, mode:str) -> None:
        """
        Sets the ellipse mode for drawing ellipses.
        
        Args:
            mode (str): The ellipse mode to set. Can be 'center' or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['center', 'radius']:
            raise ValueError(f"Invalid ellipse mode: {mode}. Must be 'center' or 'radius'.")
        self._ellipse_mode = mode
    
    @property
    def image_mode(self) -> str:
        """
        Returns the current image mode.
        
        Returns:
            str: The current image mode ('corner', 'corners', 'center', or 'radius').
        """
        return self._image_mode
    
    @image_mode.setter
    def image_mode(self, mode:str) -> None:
        """
        Sets the image mode for drawing images.
        
        Args:
            mode (str): The image mode to set. Can be 'corner', 'corners', 'center', or 'radius'.
        
        Raises:
            ValueError: If the mode is not one of the expected values.
        """
        if mode not in ['corner', 'corners', 'center', 'radius']:
            raise ValueError(f"Invalid image mode: {mode}. Must be 'corner', 'corners', 'center', or 'radius'.")
        self._image_mode = mode
    
    
    @property
    def text_align_x(self) -> int:
        """
        Returns the current horizontal text alignment.
        
        Returns:
            int: The current horizontal text alignment (0: left, 1: right).
        """
        return self._text_align_x
    @text_align_x.setter
    def text_align_x(self, align:int) -> None:
        """
        Sets the horizontal text alignment.
        
        Args:
            align (int): The horizontal text alignment to set (0: left, 1: right).
        
        Raises:
            ValueError: If the alignment is not 0 or 1.
        """
        if align not in [0, 1]:
            raise ValueError("text_align_x must be 0 (left) or 1 (right).")
        self._text_align_x = align
        
    @property
    def text_align_y(self) -> int:
        """
        Returns the current vertical text alignment.
        
        Returns:
            int: The current vertical text alignment (0: top, 1: bottom).
        """
        return self._text_align_y
    @text_align_y.setter
    def text_align_y(self, align:int) -> None:
        """
        Sets the vertical text alignment.
        
        Args:
            align (int): The vertical text alignment to set (0: top, 1: bottom).
        
        Raises:
            ValueError: If the alignment is not 0 or 1.
        """
        if align not in [0, 1]:
            raise ValueError("text_align_y must be 0 (top) or 1 (bottom).")
        self._text_align_y = align
    
    @property
    def text_align(self) -> tuple[int, int]:
        """
        Returns the current text alignment as a tuple (horizontal, vertical).
        
        Returns:
            tuple[int, int]: A tuple representing the horizontal and vertical text alignment.
        """
        return (self._text_align_x, self._text_align_y)
    @text_align.setter
    def text_align(self, align:tuple[int, int]) -> None:
        """
        Sets the text alignment.
        
        Args:
            align (tuple[int, int]): A tuple representing the horizontal and vertical text alignment.
        
        Raises:
            ValueError: If the alignment is not a tuple of two integers (0 or 1).
        """
        if not isinstance(align, tuple) or len(align) != 2:
            raise ValueError("text_align must be a tuple of two integers (horizontal, vertical).")
        if align[0] not in [0, 1] or align[1] not in [0, 1]:
            raise ValueError("text_align must be (0, 0), (1, 0), (0, 1), or (1, 1).")
        self._text_align_x, self._text_align_y = align
    
    @property
    def text_font(self) -> pygame.font.Font:
        """
        Returns the current font used for text rendering.
        
        Returns:
            pygame.font.Font: The current font object.
        """
        return self._text_font
    @text_font.setter
    def text_font(self, font:pygame.font.Font) -> None:
        """
        Sets the font for text rendering.
        
        Args:
            font (pygame.font.Font): The font object to set. If None, resets to the default font.
        
        Raises:
            TypeError: If the font is not a pygame.font.Font instance.
        """
        if font is not None and not isinstance(font, pygame.font.Font):
            raise TypeError("text_font must be a pygame.font.Font instance or None.")
        self._text_font = font if font is not None else pygame.font.Font(None, 36)
    
    @property
    def text_size(self) -> int:
        """
        Returns the current text size.
        
        Returns:
            int: The current text size.
        """
        return self._text_size
    @text_size.setter
    def text_size(self, size:int) -> None:
        """
        Sets the text size for rendering.
        
        Args:
            size (int): The size to set as the text size.
        
        Raises:
            ValueError: If the size is less than 1.
        """
        if not isinstance(size, int) or size < 1:
            raise ValueError("text_size must be an integer greater than or equal to 1.")
        self._text_size = size
        if self._text_font is not None:
            self._text_font = pygame.font.Font(self._text_font.get_name(), size)
    
    @property
    def text_leading(self) -> int:
        """
        Returns the current text leading (line spacing).
        
        Returns:
            int: The current text leading.
        """
        return self._text_leading
    @text_leading.setter
    def text_leading(self, leading:int) -> None:
        """
        Sets the text leading (line spacing) for rendering.
        
        Args:
            leading (int): The leading to set for text rendering.
        
        Raises:
            ValueError: If the leading is less than 0.
        """
        if not isinstance(leading, int) or leading < 0:
            raise ValueError("text_leading must be an integer greater than or equal to 0.")
        self._text_leading = leading
    
    
    @property
    def fill_color(self) -> Color:
        """
        Returns the current fill color.
        
        Returns:
            Color: The current fill color.
        """
        return self._fill_color
    @fill_color.setter
    def fill_color(self, color:Color) -> None:
        """
        Sets the fill color for drawing shapes.
        
        Args:
            color (Color): The color to set as the fill color.
        """
        if not isinstance(color, Color):
            raise TypeError("fill_color must be a Color instance.")
        self._fill_color = color
    
    @property
    def stroke_color(self) -> Color:
        """
        Returns the current stroke color.
        
        Returns:
            Color: The current stroke color.
        """
        return self._stroke_color
    @stroke_color.setter
    def stroke_color(self, color:Color) -> None:
        """
        Sets the stroke color for drawing shapes.
        
        Args:
            color (Color): The color to set as the stroke color.
        
        Raises:
            TypeError: If the color is not a Color instance.
        """
        if not isinstance(color, Color):
            raise TypeError("stroke_color must be a Color instance.")
        self._stroke_color = color
    
    @property
    def stroke_width(self) -> int:
        """
        Returns the current stroke width.
        
        Returns:
            int: The current stroke width.
        """
        return self._stroke_width
    @stroke_width.setter
    def stroke_width(self, width:int) -> None:
        """
        Sets the stroke width for drawing shapes.
        
        Args:
            width (int): The width to set as the stroke width.
        
        Raises:
            ValueError: If the width is less than 1.
        """
        if not isinstance(width, int) or width < 1:
            raise ValueError("stroke_width must be an integer greater than or equal to 1.")
        self._stroke_width = width
    
    @property
    def texture(self) -> pygame.Surface|None:
        """
        Returns the current texture used for filling shapes.
        
        Returns:
            pygame.Surface: The current texture surface, or None if no texture is set.
        """
        return self._texture
    @texture.setter
    def texture(self, texture:pygame.Surface|None) -> None:
        """
        Sets the texture for filling shapes.
        
        Args:
            texture (pygame.Surface | None): The texture surface to set, or None to remove the texture.
        
        Raises:
            TypeError: If the texture is not a pygame.Surface or None.
        """
        if texture is not None and not isinstance(texture, pygame.Surface):
            raise TypeError("texture must be a pygame.Surface or None.")
        self._gradient = None  # Reset gradient if texture is set
        self._texture = texture
    
    @property
    def gradient(self) -> tuple[Color, Color]|None:
        """
        Returns the current gradient used for filling shapes.
        
        Returns:
            tuple[Color, Color]: A tuple of two Color instances representing the gradient colors, or None if no gradient is set.
        """
        return self._gradient
    @gradient.setter
    def gradient(self, colors:tuple[Color, Color]|None) -> None:
        """
        Sets the gradient for filling shapes.
        
        Args:
            colors (tuple[Color, Color] | None): A tuple of two Color instances representing the gradient colors, or None to remove the gradient.
        
        Raises:
            TypeError: If colors is not a tuple of two Color instances or None.
        """
        if colors is not None:
            if not isinstance(colors, tuple) or len(colors) != 2:
                raise TypeError("gradient must be a tuple of two Color instances or None.")
            if not all(isinstance(c, Color) for c in colors):
                raise TypeError("Both elements of the gradient must be Color instances.")
        self._texture = None
        self._gradient = colors
    
    
    @property
    def curve_detail(self) -> int:
        """
        Returns the current detail level for bezier curves.
        
        Returns:
            int: The current detail level for bezier curves.
        """
        return self._curve_detail
    @curve_detail.setter
    def curve_detail(self, detail:int) -> None:
        """
        Sets the detail level for bezier curves.
        
        Args:
            detail (int): The detail level to set for bezier curves.
        
        Raises:
            ValueError: If the detail is less than 1.
        """
        if not isinstance(detail, int) or detail < 1:
            raise ValueError("curve_detail must be an integer greater than or equal to 1.")
        self._curve_detail = detail
    
    
    # Pixel Access
    def set_pixel(self, x:int, y:int, color:Color) -> None:
        """
        Sets the pixel at (x, y) to the specified color.
        
        Args:
            x (int): The x coordinate of the pixel.
            y (int): The y coordinate of the pixel.
            color (Color): The color to set the pixel to.
        """
        self.set_at((x, y), color.to_tuple())
    
    def get_pixel(self, x:int, y:int) -> Color:
        """
        Gets the color of the pixel at (x, y).
        
        Args:
            x (int): The x coordinate of the pixel.
            y (int): The y coordinate of the pixel.
        
        Returns:
            Color: The color of the pixel at (x, y).
        """
        return Color(*self.get_at((x, y)))
    
    
    # Colors
    def no_fill(self) -> None:
        """
        Disables the fill color for shapes.
        This means shapes will not be filled with any color when drawn.
        """
        self._fill_color = Color(0, 0, 0, 0)
    
    def no_stroke(self) -> None:
        """
        Disables the stroke color for shapes.
        This means shapes will not have an outline when drawn.
        """
        self._stroke_color = Color(0, 0, 0, 0)
        self._stroke_width = 0
    
    def no_texture(self) -> None:
        """
        Disables the texture for shapes.
        This means shapes will not be filled with any texture when drawn.
        """
        self._texture = None
    
    def no_gradient(self) -> None:
        """
        Disables the gradient for shapes.
        This means shapes will not be filled with any gradient when drawn.
        """
        self._gradient = None
    
    
    # Drawing Methods
    def background(self, color:Color) -> None:
        """
        Fills the entire surface with the specified background color.
        
        Args:
            color (Color): The color to fill the surface with.
        """
        self.fill(color.to_tuple())
    
    
    def path(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a path defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the path.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Draw the path outline
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def shape(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a shape defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Fill the shape with the current fill color
        self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the shape
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    
    def point(self, x:float, y:float) -> None:
        """
        Draws a point at the specified position.
        
        Args:
            x (float): The x coordinate where the point should be drawn.
            y (float): The y coordinate where the point should be drawn.
        """
        
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("x and y must be numbers.")
        
        # Transform the point using the last transformation matrix
        transformed_point = self.last_transform @ Vector2D(x, y)
        
        # Draw the point as a small circle
        self._shape_outline(
            [transformed_point, transformed_point],
            self._stroke_color,
            self._stroke_width
        )
    
    def line(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws a line from (x1, y1) to (x2, y2).
        
        Args:
            x1 (float): The x coordinate of the start point.
            y1 (float): The y coordinate of the start point.
            x2 (float): The x coordinate of the end point.
            y2 (float): The y coordinate of the end point.
        """
        
        if not all(isinstance(coord, (int, float)) for coord in [x1, y1, x2, y2]):
            raise TypeError("All coordinates must be numbers.")
        
        # Transform points using the last transformation matrix
        start_point = self.last_transform @ Vector2D(x1, y1)
        end_point = self.last_transform @ Vector2D(x2, y2)
        
        # Draw the line outline
        self._shape_outline([start_point, end_point], self._stroke_color, self._stroke_width)
    
    def bezier(self, points:Iterable[Vector2D]) -> None:
        """
        Draws a Bezier curve defined by a list of Vector2D points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the Bezier curve.
            steps (int, optional): The number of steps to use for drawing the curve. Defaults to 100.
        """
        
        if not isinstance(points, Iterable):
            raise TypeError("points must be an iterable of Vector2D instances.")
        
        if len(points) < 2:
            raise ValueError("At least two points are required to draw a Bezier curve.")
        
        # Transform points using the last transformation matrix
        transformed_points = [self.last_transform @ p for p in points]
        
        # Calculate Bezier curve points
        bezier_points = []
        for t in np.linspace(0, 1, self._curve_detail):
            ps = transformed_points
            for i, p in enumerate(transformed_points):
                for j, q in enumerate(transformed_points[:len(transformed_points) - 1 - i]):
                    ps[i] = ps[i] * (1 - t) + ps[i + 1] * t
            bezier_points.append(ps[0])
        
        # Draw the Bezier curve outline
        self._shape_outline(bezier_points, self._stroke_color, self._stroke_width)
    
    def cubic_bezier(self, x1:float, y1:float, x2:float, y2:float, x3:float, y3:float, x4:float, y4:float) -> None:
        """
        Draws a cubic Bezier curve defined by four control points.
        
        Args:
            x1 (float): The x coordinate of the first control point.
            y1 (float): The y coordinate of the first control point.
            x2 (float): The x coordinate of the second control point.
            y2 (float): The y coordinate of the second control point.
            x3 (float): The x coordinate of the third control point.
            y3 (float): The y coordinate of the third control point.
            x4 (float): The x coordinate of the fourth control point.
            y4 (float): The y coordinate of the fourth control point.
        """
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(x1, y1),
            self.last_transform @ Vector2D(x2, y2),
            self.last_transform @ Vector2D(x3, y3),
            self.last_transform @ Vector2D(x4, y4)
        ]
        
        # Calculate cubic Bezier curve points
        bezier_points = []
        for t in np.linspace(0, 1, self._curve_detail):
            u = 1 - t
            point = u**3 * transformed_points[0] + \
                    3*u**2*t * transformed_points[1] + \
                    3*u*t**2 * transformed_points[2] + \
                    t**3 * transformed_points[3]
            bezier_points.append(point)
        
        # Draw the cubic Bezier curve outline
        self._shape_outline(bezier_points, self._stroke_color, self._stroke_width)
    
    def triangle(self, x1:float, y1:float, x2:float, y2:float, x3:float, y3:float) -> None:
        """
        Draws a triangle defined by three points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
            x3 (float): The x coordinate of the third point.
            y3 (float): The y coordinate of the third point.
        """
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(x1, y1),
            self.last_transform @ Vector2D(x2, y2),
            self.last_transform @ Vector2D(x3, y3)
        ]
        
        # Fill the triangle with the current fill color
        self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the triangle
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def rectangle(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws a rectangle defined by two points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
        """
        
        # Convert coordinates based on the current rectangle mode
        coords = self._coordinates(self._rect_mode, x1, y1, x2, y2)
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(coords[0], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1] + coords[3]),
            self.last_transform @ Vector2D(coords[0], coords[1] + coords[3])
        ]
        
        # Fill the rectangle with the current fill color
        self._shape_fill(transformed_points, self._fill_color)
        
        # Draw the outline of the rectangle
        self._shape_outline(transformed_points, self._stroke_color, self._stroke_width)
    
    def ellipse(self, x1:float, y1:float, x2:float, y2:float) -> None:
        """
        Draws an ellipse defined by two points.
        
        Args:
            x1 (float): The x coordinate of the first point.
            y1 (float): The y coordinate of the first point.
            x2 (float): The x coordinate of the second point.
            y2 (float): The y coordinate of the second point.
        """
        
        # Convert coordinates based on the current ellipse mode
        left, top, width, height = self._coordinates(self._ellipse_mode, x1, y1, x2, y2)
        
        center = self.last_transform @ Vector2D(left + width / 2, top + height / 2)
        vert_vector = self.last_transform @ Vector2D(0, height / 2)
        horiz_vector = self.last_transform @ Vector2D(width / 2, 0)
        
        # Calculate the points of the ellipse
        ellipse_points = []
        for angle in np.linspace(0, 2 * np.pi, self._curve_detail):
            point = center + vert_vector * np.sin(angle) + horiz_vector * np.cos(angle)
            ellipse_points.append(point)
            
        # Fill the ellipse with the current fill color
        self._shape_fill(ellipse_points, self._fill_color)
        
        # Draw the outline of the ellipse
        self._shape_outline(ellipse_points, self._stroke_color, self._stroke_width)
    
    def image(self, image:pygame.Surface, x:float, y:float) -> None:
        """
        Draws an image on the surface at the specified position.
        
        Args:
            image (pygame.Surface): The image to draw.
            x (float): The x coordinate where the image should be drawn.
            y (float): The y coordinate where the image should be drawn.
        """
        
        if not isinstance(image, pygame.Surface):
            raise TypeError("image must be a pygame.Surface instance.")
        
        # Convert coordinates based on the current image mode
        coords = self._coordinates(self._image_mode, x, y, image.get_width(), image.get_height())
        
        # Transform points using the last transformation matrix
        transformed_points = [
            self.last_transform @ Vector2D(coords[0], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1]),
            self.last_transform @ Vector2D(coords[0] + coords[2], coords[1] + coords[3]),
            self.last_transform @ Vector2D(coords[0], coords[1] + coords[3])
        ]
        
        # Draw the image at the transformed position
        self._shape_texture(transformed_points, image, uvs=[
            Vector2D(0, 0),
            Vector2D(1, 0),
            Vector2D(1, 1),
            Vector2D(0, 1)
        ])
    
    def arc (self, x:float, y:float, radius:float, start_angle:float, end_angle:float) -> None:
        """
        Draws an arc defined by a center point, radius, and start and end angles.
        
        Args:
            x (float): The x coordinate of the center of the arc.
            y (float): The y coordinate of the center of the arc.
            radius (float): The radius of the arc.
            start_angle (float): The starting angle of the arc in radians.
            end_angle (float): The ending angle of the arc in radians.
        """
        
        if not all(isinstance(coord, (int, float)) for coord in [x, y, radius, start_angle, end_angle]):
            raise TypeError("x, y, radius, start_angle, and end_angle must be numbers.")
        
        # Transform the center point using the last transformation matrix
        center = self.last_transform @ Vector2D(x, y)
        
        # Calculate points for the arc
        arc_points = []
        for angle in np.linspace(start_angle, end_angle, self._curve_detail):
            point = center + Vector2D(radius * np.cos(angle), radius * np.sin(angle))
            arc_points.append(point)
        
        # Draw the arc outline
        self._shape_outline(arc_points, self._stroke_color, self._stroke_width)
    
    
    # Text Methods
    def text(self, text:str, x:float, y:float) -> None:
        """
        Draws text on the surface at the specified position.
        
        Args:
            text (str): The text to draw.
            x (float): The x coordinate where the text should be drawn.
            y (float): The y coordinate where the text should be drawn.
        """
        
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        
        lines = text.split('\n')
        line_offset = self._text_font.get_height() + self._text_leading
        for i, line in enumerate(lines):
            # Create a text surface
            text_surface = self._text_font.render(text, True, self._fill_color.to_tuple())
            
            # Calculate position based on alignment
            left = self._text_align_x * (self.width - text_surface.get_width())
            top = self._text_align_y * (self.height - text_surface.get_height())
            position = (x + left, y + top)
            
            transformed_points = [self.last_transform @ vector for vector in [
                Vector2D(position[0], position[1] + line_offset * i),
                Vector2D(position[0] + text_surface.get_width(), position[1] + line_offset * i),
                Vector2D(position[0] + text_surface.get_width(), position[1] + text_surface.get_height() + line_offset * i),
                Vector2D(position[0], position[1] + text_surface.get_height() + line_offset * i)
            ]]
            
            # Draw the text surface at the transformed position
            self._shape_texture(transformed_points, text_surface, uvs=[
                Vector2D(0, 0),
                Vector2D(1, 0),
                Vector2D(1, 1),
                Vector2D(0, 1)
            ])
    
    @staticmethod
    def list_fonts() -> list[str]:
        """
        Lists all available fonts in Pygame.
        
        Returns:
            list[str]: A list of font names available in Pygame.
        """
        return pygame.font.get_fonts()
    
    
    # Transformations
    def push_matrix(self) -> None:
        """
        Saves the current transformation matrix onto the stack.
        This allows for nested transformations.
        """
        self._transforms.append(self._transfroms[-1].copy())
        return self._MatrixContext(self)
    
    def pop_matrix(self) -> None:
        """
        Restores the last transformation matrix from the stack.
        This undoes the last push_matrix call.
        
        Raises:
            IndexError: If there are no matrices to pop.
        """
        if len(self._transforms) <= 1:
            raise IndexError("No transformation matrix to pop.")
        self._transforms.pop()
    
    def reset_matrix(self) -> None:
        """
        Resets the transformation matrix to the identity matrix.
        This clears all transformations applied so far.
        """
        self._transforms = [Matrix2D.identity()]
    
    def apply_matrix(self, matrix:Matrix2D) -> None:
        """
        Applies a transformation matrix to the current transformation stack.
        
        Args:
            matrix (Matrix2D): The transformation matrix to apply.
        
        Raises:
            TypeError: If the matrix is not an instance of Matrix2D.
        """
        if not isinstance(matrix, Matrix2D):
            raise TypeError("matrix must be an instance of Matrix2D.")
        self.last_transform = self.last_transform @ matrix
    
    def translate(self, x:float, y:float) -> None:
        """
        Translates the current transformation matrix by (x, y).
        
        Args:
            x (float): The x translation amount.
            y (float): The y translation amount.
        """
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise TypeError("x and y must be numbers.")
        self.last_transform = self.last_transform.translate(x, y)
    
    def rotate(self, angle:float) -> None:
        """
        Rotates the current transformation matrix by the specified angle in radians.
        
        Args:
            angle (float): The angle in radians to rotate the matrix.
        
        Raises:
            TypeError: If the angle is not a number.
        """
        if not isinstance(angle, (int, float)):
            raise TypeError("angle must be a number.")
        self.last_transform = self.last_transform.rotate(angle)
    
    def scale(self, sx:float, sy:float) -> None:
        """
        Scales the current transformation matrix by (sx, sy).
        
        Args:
            sx (float): The x scale factor.
            sy (float): The y scale factor.
        
        Raises:
            TypeError: If sx or sy is not a number.
        """
        if not isinstance(sx, (int, float)) or not isinstance(sy, (int, float)):
            raise TypeError("sx and sy must be numbers.")
        self.last_transform = self.last_transform.scale(sx, sy)
    
    class _MatrixContext:
        def __init__(self, graphics:Graphics):
            """
            Context manager for applying transformations to the Graphics object.
            
            Args:
                graphics (Graphics): The Graphics object to apply transformations to.
            """
            self.graphics = graphics
        def __enter__(self):
            """
            Enters the context manager, saving the current transformation matrix.
            """
            self.graphics.push_matrix()
            return self.graphics
        def __exit__(self, exc_type, exc_value, traceback):
            """
            Exits the context manager, restoring the last transformation matrix.
            
            Args:
                exc_type: The exception type, if any.
                exc_value: The exception value, if any.
                traceback: The traceback object, if any.
            """
            self.graphics.pop_matrix()
    
    
    # Abstractions
    def _shape_outline(self, points:Iterable[Vector2D], color:Color=None, width:int=1) -> None:
        """
        Draws an outline of a shape defined by a list of points.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
            color (Color, optional): The color of the outline. Defaults to the current stroke color.
            width (int, optional): The width of the outline. Defaults to 1.
        """
        
        if color is None:
            color = self._stroke_color
        
        pygame.draw.lines(self, color.to_tuple(), True, [p.to_tuple() for p in points], width)
    
    def _shape_fill(self, points:Iterable[Vector2D], color:Color=None) -> None:
        """
        Fills a shape defined by a list of points with the specified color.
        
        Args:
            points (Iterable[Vector2D]): An iterable of Vector2D points defining the shape.
            color (Color, optional): The color to fill the shape with. Defaults to the current fill color.
        """
        
        if color is None:
            color = self._fill_color
        pygame.draw.polygon(self, color.to_tuple(), [p.to_tuple() for p in points])
    
    def _shape_texture(self, points:Iterable[Vector2D], texture:pygame.Surface, uvs:Iterable[Vector2D]=None) -> None:
        # TODO: Test
        """
        Fills a shape defined by a list of points with a texture.

        Args:
            points (Iterable[Vector2D]): _description_
            texture (pygame.Surface): _description_
            uvs (Iterable[Vector2D], optional): _description_. Defaults to None.
        """
        mask_surface = pygame.Surface(texture.get_size(), pygame.SRCALPHA)
        mask_surface.fill((0, 0, 0, 0))  # Transparent background

        # Scale UV coordinates to texture size
        scaled_points = [
            (uv[0] * texture.get_width(), uv[1] * texture.get_height())
            for uv in uvs
        ]

        # Draw the polygon on the mask surface
        pygame.draw.polygon(mask_surface, (255, 255, 255, 255), scaled_points)

        # Blit the texture onto the mask surface
        mask_surface.blit(texture, (0, 0))

        # Create a final surface for the polygon
        final_surface = pygame.Surface((self.get_width(), self.get_height()), pygame.SRCALPHA)
        pygame.draw.polygon(final_surface, (255, 255, 255, 255), points)
        
        # Apply the mask to the polygon
        final_surface.blit(mask_surface, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        
        # Blit the final surface onto the main surface
        self.blit(final_surface, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    # TODO: Implement more context managers for transformations and drawing
# TODO: Implement Shape drawing logic in Graphics

class Animation:
    # Static class with nested classes like Ease, Formula, Keyframe, Envelope and Timeline.
    @staticmethod
    def animate(t, ease, formula):
        """
        Animates a value based on time, easing function, and formula.
        Args:
            t (float): The time value between 0 and 1.
            ease (Ease): The easing function to apply.
            formula (Formula): The mathematical formula to apply.
        Returns:
            float: The animated value based on the easing function and formula.
        """
        return ease.value(formula.value(ease.time(t)), t)
class Ease:
    def __init__(self):
        """
        Static class for easing functions.
        """
        pass
    def time(self, t:float) -> float:
        """
        Argument for Formula.value().
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The eased value.
        """
        return t
    def value(self, t:float, v:float) -> float:
        """
        Final part of the equation.
        
        Args:
            t (float): The time value between 0 and 1.
            v (float): The value to ease.
        
        Returns:
            float: The eased value.
        """
        return v
Animation.Ease = Ease; Ease = None
class In(Animation.Ease):
    """
    Easing function for ease-in.
    """
    def time(self, t:float) -> float:
        return t
    def value(self, t:float, v:float) -> float:
        return v
Animation.Ease.In = In; In = None
class Out(Animation.Ease):
    """
    Easing function for ease-out.
    """
    def time(self, t:float) -> float:
        return 1 - t
    def value(self, t:float, v:float) -> float:
        return 1 - v
Animation.Ease.Out = Out; Out = None
class InOut(Animation.Ease):
    """
    Easing function for ease-in-out.
    """
    def time(self, t:float) -> float:
        return 2 * t if t < 0.5 else 2 * (1 - t)
    def value(self, t:float, v:float) -> float:
        return v if t < 0.5 else 1 - v
Animation.Ease.InOut = InOut; InOut = None
class OutIn(Animation.Ease):
    """
    Easing function for ease-out-in.
    """
    def time(self, t:float) -> float:
        return 1 - (2 * t if t < 0.5 else 2 * (1 - t))
    def value(self, t:float, v:float) -> float:
        return 1 - v if t < 0.5 else v
Animation.Ease.OutIn = OutIn; OutIn = None
class Formula:
    def __init__(self):
        """
        Static class for mathematical formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a mathematical formula to the time value.
        Args:
            t (float): The time value between 0 and 1.
        Returns:
            float: The calculated value.
        """
        return t
Animation.Formula = Formula; Formula = None
class Polynomial(Animation.Formula):
    """
    Polynomial formula for easing.
    """
    def __init__(self, degree:int=2):
        """
        Initializes the polynomial formula with a degree.
        
        Args:
            degree (int): The degree of the polynomial. Defaults to 2.
        """
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be an integer greater than or equal to 1.")
        self.degree = degree
    def value(self, t:float) -> float:
        """
        Applies the polynomial formula to the time value.
        Args:
            t (float): The time value between 0 and 1.
        Returns:
            float: The calculated value based on the polynomial formula.
        """
        if not isinstance(t, (int, float)):
            raise TypeError("t must be a number.")
        return t ** self.degree
Animation.Formula.Polynomial = Polynomial; Polynomial = None
class Exponential(Animation.Formula):
    """
    Exponential formula for easing.
    """
    def __init__(self, base:float=2.0):
        """
        Initializes the exponential formula with a base.
        
        Args:
            base (float): The base of the exponential function. Defaults to 2.0.
        """
        if not isinstance(base, (int, float)) or base <= 0:
            raise ValueError("base must be a positive number.")
        self.base = base
    def value(self, t:float) -> float:
        """
        Applies the exponential formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the exponential formula.
        """
        if not isinstance(t, (int, float)):
            raise TypeError("t must be a number.")
        return self.base ** t
Animation.Formula.Exponential = Exponential; Exponential = None
class Trigoniometric(Animation.Formula):
    def __init__(self):
        """
        Static class for trigonometric formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a trigonometric formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the trigonometric formula.
        """
        return np.sin(t * np.pi * 2)
Animation.Formula.Trigoniometric = Trigoniometric; Trigoniometric = None
class Circular(Animation.Formula):
    def __init__(self):
        """
        Static class for circular formulas.
        """
        pass
    def value(self, t:float) -> float:
        """
        Applies a circular formula to the time value.
        
        Args:
            t (float): The time value between 0 and 1.
        
        Returns:
            float: The calculated value based on the circular formula.
        """
        return np.sqrt(1 - (t - 1) ** 2)
Animation.Formula.Circular = Circular; Circular = None
class Keyframe:
    def __init__(self, time:float, value:float, ease:Animation.Ease=None):
        """
        Initializes a keyframe with a time, value, and optional easing function.
        
        Args:
            time (float): The time from the last keyframe.
            value (float): The value of the keyframe.
            ease (Animation.Ease, optional): The easing function to apply. Defaults to None.
        """
        if not isinstance(time, (int, float)):
            raise TypeError("time must be a number.")
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.time = time
        self.value = value
        self.ease = ease if ease is not None else Animation.Ease()
Animation.Keyframe = Keyframe; Keyframe = None
class Envelope:
    def __init__(self, keyframes:list[Animation.Keyframe]=[], delay:float=0.0):
        """
        Initializes an envelope with a list of keyframes.
        
        Args:
            keyframes (list[Animation.Keyframe]): A list of keyframes defining the envelope.
        """
        if not isinstance(keyframes, list) or not all(isinstance(kf, Animation.Keyframe) for kf in keyframes):
            raise TypeError("keyframes must be a list of Animation.Keyframe instances.")
        self.keyframes = keyframes
        self.delay = delay
    def add_keyframe(self, keyframe:Animation.Keyframe) -> None:
        """
        Adds a keyframe to the envelope.
        Args:
            keyframe (Animation.Keyframe): The keyframe to add.
        Raises:
            TypeError: If keyframe is not an instance of Animation.Keyframe.
        """
        if not isinstance(keyframe, Animation.Keyframe):
            raise TypeError("keyframe must be an instance of Animation.Keyframe.")
        self.keyframes.append(keyframe)
    def remove_keyframe(self, keyframe:Animation.Keyframe) -> None:
        """
        Removes a keyframe from the envelope.
        Args:
            keyframe (Animation.Keyframe): The keyframe to remove.
        Raises:
            TypeError: If keyframe is not an instance of Animation.Keyframe.
        """
        if not isinstance(keyframe, Animation.Keyframe):
            raise TypeError("keyframe must be an instance of Animation.Keyframe.")
        self.keyframes.remove(keyframe)
    def __len__(self) -> int:
        """
        Returns the number of keyframes in the envelope.
        
        Returns:
            int: The number of keyframes.
        """
        return len(self.keyframes)
    def __getitem__(self, index:int) -> Animation.Keyframe:
        """
        Gets a keyframe by index.
        
        Args:
            index (int): The index of the keyframe to retrieve.
        
        Returns:
            Animation.Keyframe: The keyframe at the specified index.
        """
        if not isinstance(index, int):
            raise TypeError("index must be an integer.")
        return self.keyframes[index]
    def __iter__(self):
        """
        Returns an iterator over the keyframes in the envelope.
        
        Returns:
            Iterator[Animation.Keyframe]: An iterator over the keyframes.
        """
        return iter(self.keyframes)
    def duration(self) -> float:
        """
        Calculates the total duration of the envelope based on the keyframes.
        
        Returns:
            float: The total duration of the envelope.
        """
        if not self.keyframes:
            return 0.0
        return self.keyframes[-1].time
    def value_at(self, time:float) -> float:
        """
        Gets the value at a specific time based on the keyframes.
        
        Args:
            time (float): The time to get the value for.
        
        Returns:
            float: The value at the specified time.
        """
        if not isinstance(time, (int, float)):
            raise TypeError("time must be a number.")
        if not self.keyframes:
            return 0.0
        
        # Find the two keyframes surrounding the time
        for i in range(len(self.keyframes) - 1):
            if self.keyframes[i].time <= time <= self.keyframes[i + 1].time:
                kf1 = self.keyframes[i]
                kf2 = self.keyframes[i + 1]
                t = (time - kf1.time) / (kf2.time - kf1.time)
                return Animation.Ease().value(t, kf1.value + (kf2.value - kf1.value) * t)
        
        # If time is after the last keyframe, return the last keyframe's value
        if time >= self.keyframes[-1].time:
            return self.keyframes[-1].value
        # If time is before the first keyframe, return the first keyframe's value
        return self.keyframes[0].value
    def copy(self) -> 'Animation.Envelope':
        """
        Creates a copy of the envelope.
        
        Returns:
            Animation.Envelope: A new envelope with the same keyframes and delay.
        """
        return Animation.Envelope(self.keyframes.copy(), self.delay)
Animation.Envelope = Envelope; Envelope = None
class LiveValue:
    def __init__(self, value:float|int=0.0):
        """
        Initializes a live value that can be animated.
        
        Args:
            value (float|int, optional): The initial value. Defaults to 0.0.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.value = value
        self.base_value = value
        self.envelopes = []
    
    @property
    def current_value(self) -> float:
        """
        Gets the current value of the live value, modified by any active envelopes.
        
        Returns:
            float: The current value.
        """
        self.update()
        return self.value
    
    def update(self) -> None:
        """
        Updates the live value based on the active envelopes.
        This method should be called regularly to ensure the value is updated.
        """
        value = 0.0
        for envelope in self.envelopes:
            if time.time() - envelope.delay > envelope.duration():
                self.base_value += envelope.value_at(time.time())
            else:
                value += envelope.value_at(time.time() - envelope.delay)
        self.value = self.base_value + value
    def animate_now(self, envelope:Animation.Envelope) -> None:
        """
        Immediately applies an envelope to the live value.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        envelope.delay = time.time()
        self.envelopes.append(envelope.copy())
        self.update()
    def animate_after(self, envelope:Animation.Envelope, delay:float) -> None:
        """
        Applies an envelope to the live value after a specified delay.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
            delay (float): The delay in seconds before applying the envelope.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        if not isinstance(delay, (int, float)):
            raise TypeError("delay must be a number.")
        envelope.delay = time.time() + delay
        self.envelopes.append(envelope.copy())
    def animate_end(self, envelope:Animation.Envelope) -> None:
        """
        Ends the animation of an envelope.
        
        Args:
            envelope (Animation.Envelope): The envelope to end.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        # Calculate the time at which the all envelopes end
        end_time = time.time()
        for env in self.envelopes:
            if env.duration() + env.delay > end_time:
                end_time = env.duration() + env.delay
        envelope.delay = end_time
        self.envelopes.append(envelope.copy())
    def animate(self, envelope:Animation.Envelope) -> None:
        """
        Applies an envelope to the live value.
        
        Args:
            envelope (Animation.Envelope): The envelope to apply.
        """
        if not isinstance(envelope, Animation.Envelope):
            raise TypeError("envelope must be an instance of Animation.Envelope.")
        self.envelopes.append(envelope.copy())
        self.update()
    
    def set_immediate_value(self, value:float|int) -> None:
        """
        Sets the live value immediately without animation.
        
        Args:
            value (float|int): The value to set.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        self.value = value
        self.base_value = value
    def set_ending_value(self, value:float|int) -> None:
        """
        Sets the live value to an ending value, clearing all envelopes.
        
        Args:
            value (float|int): The ending value to set.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("value must be a number.")
        # Calculate final value
        final_value = self.get_ending_value()
        self.base_value = final_value - self.value + value
    def get_ending_value(self) -> float:
        """
        Gets the ending value of the live value, considering all envelopes.
        
        Returns:
            float: The ending value.
        """
        final_value = self.base_value
        for envelope in self.envelopes:
            final_value += envelope.value_at(envelope.duration())
        return final_value
Animation.LiveValue = LiveValue; LiveValue = None

# YuiElement
class Yui:
    def __init__(self, parent:Yui):
        # --- Transform ---
        self._x = 0
        self._y = 0
        self._r = 0
        self._sx = 1
        self._sy = 1
        self._ax = 0
        self._ay = 0
        
        # --- Matrix Cache & Flags ---
        self._local_matrix = Matrix2D.identity()
        self._world_matrix = Matrix2D.identity()
        self._local_inverted_matrix = Matrix2D.identity()
        self._world_inverted_matrix = Matrix2D.identity()
        self._needs_local_matrix_update = True
        self._needs_world_matrix_update = True

        # --- Graphics ---
        self._graphics = None
        self._needs_graphics_rebuild = True
        self._uses_graphics = False
        self._width = 1
        self._height = 1
        
        # --- Hierarchy ---
        self._parent = None
        self._children = []
        
        # --- Flags ---
        self._destroyed = False
        self._visible = True
        self._enabled = True

        # --- Debug ---
        self._draw_time_self = 0.0
        self._draw_time_subtree = 0.0

        self.set_parent(parent)
    
    def __getitem__(self, key:int|slice) -> 'Yui':
        """
        Gets a child Yui element by index or slice.
        
        Args:
            key (int|slice): The index or slice to access.
        
        Returns:
            Yui: The child Yui element(s).
        """
        if isinstance(key, int):
            return self._children[key]
        elif isinstance(key, slice):
            return self._children[key]
        else:
            raise TypeError("key must be an integer or slice.")
    def __setitem__(self, key:int, value:'Yui') -> None:
        raise NotImplementedError("Setting children directly is not supported. Use add_child() instead.")
    def __delitem__(self, key:int|slice) -> None:
        """
        Deletes a child Yui element by index.
        Args:
            key (int|slice): The index or slice to delete.
        Raises:
            TypeError: If key is not an integer or slice.
        """
        if isinstance(key, int):
            if 0 <= key < len(self._children):
                child = self._children[key]
                child.set_parent(None)
                del self._children[key]
            else:
                raise IndexError("Child index out of range.")
        elif isinstance(key, slice):
            for child in self._children[key]:
                child.set_parent(None)
            del self._children[key]
        else:
            raise TypeError("key must be an integer or slice.")
    def __len__(self) -> int:
        """
        Returns the number of child Yui elements.
        
        Returns:
            int: The number of children.
        """
        return len(self._children)
    
    @property
    def x(self) -> float:
        """
        Gets the x coordinate of this Yui element.
        
        Returns:
            float: The x coordinate.
        """
        return self._x
    @x.setter
    def x(self, value:float):
        """
        Sets the x coordinate of this Yui element.
        
        Args:
            value (float): The new x coordinate.
        """
        last_x = self.x
        if not isinstance(value, (int, float)):
            raise TypeError("x must be a number.")
        self._x = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(last_x, self.y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def y(self) -> float:
        """
        Gets the y coordinate of this Yui element.
        
        Returns:
            float: The y coordinate.
        """
        return self._y
    @y.setter
    def y(self, value:float):
        """
        Sets the y coordinate of this Yui element.
        
        Args:
            value (float): The new y coordinate.
        """
        last_y = self.y
        if not isinstance(value, (int, float)):
            raise TypeError("y must be a number.")
        self._y = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, last_y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def r(self) -> float:
        """
        Gets the rotation of this Yui element in radians.
        
        Returns:
            float: The rotation in radians.
        """
        return self._r
    @r.setter
    def r(self, value:float):
        """
        Sets the rotation of this Yui element in radians.
        
        Args:
            value (float): The new rotation in radians.
        """
        last_r = self.r
        if not isinstance(value, (int, float)):
            raise TypeError("r must be a number.")
        self._r = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, last_r, self.sx, self.sy, self.ax, self.ay)
    @property
    def sx(self) -> float:
        """
        Gets the x scale factor of this Yui element.
        
        Returns:
            float: The x scale factor.
        """
        return self._sx
    @sx.setter
    def sx(self, value:float):
        """
        Sets the x scale factor of this Yui element.
        
        Args:
            value (float): The new x scale factor.
        """
        last_sx = self.sx
        if not isinstance(value, (int, float)):
            raise TypeError("sx must be a number.")
        self._sx = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, last_sx, self.sy, self.ax, self.ay)
    @property
    def sy(self) -> float:
        """
        Gets the y scale factor of this Yui element.
        
        Returns:
            float: The y scale factor.
        """
        return self._sy
    @sy.setter
    def sy(self, value:float):
        """
        Sets the y scale factor of this Yui element.
        
        Args:
            value (float): The new y scale factor.
        """
        last_sy = self.sy
        if not isinstance(value, (int, float)):
            raise TypeError("sy must be a number.")
        self._sy = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, last_sy, self.ax, self.ay)
    @property
    def ax(self) -> float:
        """
        Gets the x anchor point of this Yui element.
        
        Returns:
            float: The x anchor point.
        """
        return self._ax
    @ax.setter
    def ax(self, value:float):
        """
        Sets the x anchor point of this Yui element.
        
        Args:
            value (float): The new x anchor point.
        """
        last_ax = self.ax
        if not isinstance(value, (int, float)):
            raise TypeError("ax must be a number.")
        self._ax = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, last_ax, self.ay)
    @property
    def ay(self) -> float:
        """
        Gets the y anchor point of this Yui element.
        
        Returns:
            float: The y anchor point.
        """
        return self._ay
    @ay.setter
    def ay(self, value:float):
        """
        Sets the y anchor point of this Yui element.
        
        Args:
            value (float): The new y anchor point.
        """
        last_ay = self.ay
        if not isinstance(value, (int, float)):
            raise TypeError("ay must be a number.")
        self._ay = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, self.ax, last_ay)
    @property
    def position(self) -> Vector2D:
        """
        Gets the position of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The position of the element.
        """
        return Vector2D(self._x, self._y)
    @position.setter
    def position(self, value:Vector2D|tuple):
        """
        Sets the position of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new position as a Vector2D or a tuple (x, y).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_position = self.position
        if isinstance(value, Vector2D):
            self._x, self._y = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._x, self._y = value
        else:
            raise TypeError("position must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(last_position.x, last_position.y, self.r, self.sx, self.sy, self.ax, self.ay)
    @property
    def rotation(self) -> float:
        """
        Gets the rotation of this Yui element in radians.
        
        Returns:
            float: The rotation in radians.
        """
        return self._r
    @rotation.setter
    def rotation(self, value:float):
        """
        Sets the rotation of this Yui element in radians.
        
        Args:
            value (float): The new rotation in radians.
        
        Raises:
            TypeError: If value is not a number.
        """
        last_rotation = self.rotation
        if not isinstance(value, (int, float)):
            raise TypeError("rotation must be a number.")
        self._r = value
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, last_rotation, self.sx, self.sy, self.ax, self.ay)
    @property
    def scale(self) -> Vector2D:
        """
        Gets the scale of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The scale of the element.
        """
        return Vector2D(self._sx, self._sy)
    @scale.setter
    def scale(self, value:Vector2D|tuple):
        """
        Sets the scale of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new scale as a Vector2D or a tuple (sx, sy).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_scale = self.scale
        if isinstance(value, Vector2D):
            self._sx, self._sy = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._sx, self._sy = value
        else:
            raise TypeError("scale must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, last_scale.x, last_scale.y, self.ax, self.ay)
    @property
    def anchor(self) -> Vector2D:
        """
        Gets the anchor point of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The anchor point of the element.
        """
        return Vector2D(self._ax, self._ay)
    @anchor.setter
    def anchor(self, value:Vector2D|tuple):
        """
        Sets the anchor point of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new anchor point as a Vector2D or a tuple (ax, ay).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        last_anchor = self.anchor
        if isinstance(value, Vector2D):
            self._ax, self._ay = value.x, value.y
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._ax, self._ay = value
        else:
            raise TypeError("anchor must be a Vector2D or a tuple of two numbers.")
        self._needs_local_matrix_update = True
        self.on_transform_changed(self.x, self.y, self.r, self.sx, self.sy, last_anchor.x, last_anchor.y)
    
    # --- Matrix ---
    def _update_matrices(self):
        changed = False
        last_local_matrix = self._local_matrix
        last_local_matrix_inverted = self._local_inverted_matrix
        last_world_matrix = self._world_matrix
        last_world_inverted_matrix = self._world_inverted_matrix

        if self._needs_local_matrix_update:
            self._local_matrix = Matrix2D.identity()
            self._local_matrix.translate(self._x, self._y)
            self._local_matrix.rotate(self._r)
            self._local_matrix.scale(self._sx, self._sy)
            self._local_matrix.translate(-self._ax, -self._ay)
            self._local_inverted_matrix = self._local_matrix.invert()
            self._needs_local_matrix_update = False
            changed = True
        if self._needs_world_matrix_update:
            if self._parent is None:
                self._world_matrix = self._local_matrix
                self._world_inverted_matrix = self._local_inverted_matrix
            else:
                self._parent._update_matrices()
                self._world_matrix = self._parent.world_matrix.multiply(self._local_matrix)
                self._world_inverted_matrix = self._world_matrix.invert()
            changed = True
            self._needs_world_matrix_update = False
        if self.on_matrix_updated(self.)

    def _has_ancestor_requested_world_matrix_update(self) -> bool:
        """
        Checks if any ancestor has requested a world matrix update.
        
        Returns:
            bool: True if an ancestor has requested a world matrix update, False otherwise.
        """
        current = self._parent
        while current is not None:
            if current._needs_world_matrix_update:
                return True
            current = current._parent
        return False
    @property
    def local_matrix(self) -> Matrix2D:
        """
        Gets the local transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The local transformation matrix.
        """
        if self._needs_local_matrix_update:
            self._update_matrices()
        return self._local_matrix
    @property
    def world_matrix(self) -> Matrix2D:
        """
        Gets the world transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The world transformation matrix.
        """
        if self._needs_world_matrix_update or self._has_ancestor_requested_world_matrix_update:
            self._update_matrices()
        return self._world_matrix
    @property
    def local_inverted_matrix(self) -> Matrix2D:
        """
        Gets the inverted local transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The inverted local transformation matrix.
        """
        if self._needs_local_matrix_update:
            self._update_matrices()
        return self._local_inverted_matrix
    @property
    def world_inverted_matrix(self) -> Matrix2D:
        """
        Gets the inverted world transformation matrix of this Yui element.
        
        Returns:
            Matrix2D: The inverted world transformation matrix.
        """
        if self._needs_world_matrix_update or self._has_ancestor_requested_world_matrix_update:
            self._update_matrices()
        return self._world_inverted_matrix

    def to_local_matrix(self, point:Vector2D|tuple) -> Vector2D|tuple:
        """
        Converts a point from world coordinates to local coordinates.
        
        Args:
            point (Vector2D|tuple): The point in world coordinates.
        
        Returns:
            Vector2D|tuple: The point in local coordinates.
        """
        if isinstance(point, Vector2D):
            return self.world_inverted_matrix.transform(point)
        elif isinstance(point, tuple) and len(point) == 2:
            tp = self.world_inverted_matrix.transform(Vector2D(*point))
            return (tp.x, tp.y)
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")
    def to_world_matrix(self, point:Vector2D|tuple) -> Vector2D|tuple:
        """
        Converts a point from local coordinates to world coordinates.
        
        Args:
            point (Vector2D|tuple): The point in local coordinates.
        
        Returns:
            Vector2D|tuple: The point in world coordinates.
        """
        if isinstance(point, Vector2D):
            return self.world_matrix.transform(point)
        elif isinstance(point, tuple) and len(point) == 2:
            tp = self.world_matrix.transform(Vector2D(*point))
            return (tp.x, tp.y)
        else:
            raise TypeError("point must be a Vector2D or a tuple of two numbers.")

    @property
    def local_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the local bounds of this Yui element.
        
        Returns:
            tuple: A tuple (x, y, width, height) representing the local bounds.
        """
        return (self._x - self._ax * self._sx, self._y - self._ay * self._sy, self._width * self._sx, self._height * self._sy)
    @property
    def world_bounds(self) -> tuple[float, float, float, float]:
        """
        Gets the world bounds of this Yui element.
        
        Returns:
            tuple: A tuple (x, y, width, height) representing the world bounds.
        """
        if self._needs_world_matrix_update or self._has_ancestor_requested_world_matrix_update:
            self._update_matrices()
        local_bounds = self.local_bounds
        transformed = self.world_matrix.transform_rect(local_bounds)
        return (transformed.x, transformed.y, transformed.width, transformed.height)

    # --- Hierarchy ---
    @property
    def parent(self) -> 'Yui':
        """
        Gets the parent of this Yui element.
        
        Returns:
            Yui: The parent element, or None if this is a root element.
        """
        return self._parent
    @property
    def children(self) -> list['Yui']:
        """
        Gets the children of this Yui element.
        
        Returns:
            list[Yui]: A list of child elements.
        """
        return self._children.copy()
    @property
    def root(self) -> 'YuiRoot':
        """
        Gets the root element of this Yui element.
        
        Returns:
            YuiRoot: The root element.
        """
        current = self
        while current._parent is not None:
            current = current._parent
        return current
    @property
    def is_root(self) -> bool:
        """
        Checks if this Yui element is the root element.
        
        Returns:
            bool: True if this is the root element, False otherwise.
        """
        return isinstance(self, YuiRoot)
    @property
    def is_leaf(self) -> bool:
        """
        Checks if this Yui element is a leaf (has no children).
        
        Returns:
            bool: True if this element has no children, False otherwise.
        """
        return len(self._children) == 0
    @property
    def depth(self) -> int:
        """
        Gets the level of this Yui element in the hierarchy.
        
        Returns:
            int: The level of the element, where 0 is the root.
        """
        level = 0
        current = self._parent
        while current is not None:
            level += 1
            current = current._parent
        return level
    @property
    def height(self) -> int:
        """
        Gets the height of this Yui element.
        
        Returns:
            float: The height of the element, which is always 0 for a base Yui element.
        """
        return 0 if self.is_leaf else max(child.height for child in self._children) + 1
    @property
    def ancestors(self) -> list['Yui']:
        """
        Gets a list of all ancestor elements, starting from the immediate parent up to the root.

        Returns:
            list[Yui]: A list of ancestor elements, ordered from closest parent to root.
        """
        ancestors = []
        current = self._parent
        while current is not None:
            ancestors.append(current)
            current = current._parent
        return ancestors
    
    def is_ancestor_of(self, other: 'Yui') -> bool:
        if other is None: return False
        return other.is_descendant_of(self)
    def is_descendant_of(self, other: 'Yui') -> bool:
        return other in self.ancestors
    
    def set_parent(self, parent: 'Yui', index:int=None):
        if self._destroyed or isinstance(self, YuiRoot) or (parent is not None and not parent.is_destroyed):
            return
        
        if parent is None: # Can't assign a null parent
            if self.parent:
                raise RuntimeError("Tried to assign a null parent on Yui init.")
            else:
                raise RuntimeError("Tried to assign a null parent on Yui parent change.")
        else:
            if parent.is_descendant_of(self): # Can only happen after init
                return
            
            if not self.parent: # Initializing
                index = max(0, min(parent.child_count, index if index else 0x7FFFFFFF))
                if not parent.can_child_be_added(self, index) or not self.can_parent_be_set(parent): # Can't be initialized with this parent, would default to None
                    raise RuntimeError("Can't assign this parent in Yui init.")
                self._parent = parent
                self._parent._children.insert(index, self)
                self.on_parent_set(None)
                self._parent.on_child_added(self, index)
            elif self._parent == parent:
                old_index = self._parent._children.index(self)
                new_index = max(0, min(self._parent.child_count - 1, index))
                if not self._parent.can_child_be_moved(self, old_index, new_index): # No change if not allowed to move
                    return
                self._parent._children.remove(self)
                self._parent._children.insert(new_index, self)
                self._parent.on_child_moved(self, old_index, new_index)
            else: # Already initialized
                old_index = self._parent._children.index(self)
                new_index = max(0, min(parent.child_count, index))
                old_parent = self._parent
                if self.can_parent_be_set(parent) and self.can_parent_be_removed(self._parent) and self._parent.can_child_be_removed(self, old_index) and parent.can_child_be_added(self, new_index):
                    self._parent._children.remove(self)
                    self._parent = parent
                    self._parent._children.insert(index, self)
                    self.on_parent_removed(old_parent)
                    old_parent.on_child_removed(self, old_index)
                    self.on_parent_set(self._parent)
                    self._parent.on_child_added(self, new_index)
    def add_child(self, child:'Yui', index:int=None) -> None:
        """
        Adds a child Yui element to this Yui element.
        
        Args:
            child (Yui): The child element to add.
            index (int, optional): The index at which to add the child. Defaults to None, which adds at the end.
        
        Raises:
            TypeError: If child is not an instance of Yui.
            RuntimeError: If the child cannot be added due to hierarchy constraints.
        """
        if not isinstance(child, Yui):
            raise TypeError("child must be an instance of Yui.")
        if child.is_destroyed:
            raise RuntimeError("Cannot add a destroyed Yui element as a child.")
        if self._destroyed or isinstance(self, YuiRoot):
            raise RuntimeError("Cannot add children to a destroyed or root Yui element.")
        
        if index is None:
            index = len(self._children)
        else:
            index = max(0, min(len(self._children), index))
        
        if not self.can_child_be_added(child, index):
            raise RuntimeError("Cannot add this child to the parent Yui element.")
        
        child.set_parent(self, index)
    def add_children(self, children:list['Yui'], index:int=None) -> None:
        """
        Adds multiple child Yui elements to this Yui element.
        
        Args:
            children (list[Yui]): A list of child elements to add.
            index (int, optional): The index at which to add the children. Defaults to None, which adds at the end.
        
        Raises:
            TypeError: If any child is not an instance of Yui.
            RuntimeError: If any child cannot be added due to hierarchy constraints.
        """
        if not isinstance(children, list) or not all(isinstance(child, Yui) for child in children):
            raise TypeError("children must be a list of Yui instances.")
        if self._destroyed or isinstance(self, YuiRoot):
            raise RuntimeError("Cannot add children to a destroyed or root Yui element.")
        
        for child in children:
            if child.is_destroyed:
                raise RuntimeError("Cannot add a destroyed Yui element as a child.")
        
        if index is None:
            index = len(self._children)
        else:
            index = max(0, min(len(self._children), index))
        
        for child in children:
            self.add_child(child, index)

    # --- Graphics ---
    @property
    def graphics(self) -> 'Graphics':
        """
        Gets the graphics object associated with this Yui element.
        
        Returns:
            Graphics: The graphics object, or None if not set.
        """
        return self._graphics
    @property
    def uses_graphics(self) -> bool:
        """
        Checks if this Yui element uses graphics.
        
        Returns:
            bool: True if the element uses graphics, False otherwise.
        """
        return self._uses_graphics
    @uses_graphics.setter
    def uses_graphics(self, value:bool):
        """
        Sets whether this Yui element uses graphics.
        
        Args:
            value (bool): True to enable graphics, False to disable.
        """
        if not isinstance(value, bool):
            raise TypeError("uses_graphics must be a boolean.")
        if self._uses_graphics != value:
            self._uses_graphics = value
            if value:
                self._needs_graphics_rebuild = True
            else:
                self._graphics = None
                self._needs_graphics_rebuild = False
    
    @property
    def width(self) -> float:
        """
        Gets the width of this Yui element.
        
        Returns:
            float: The width of the element.
        """
        return self._width
    @width.setter
    def width(self, value:float):
        """
        Sets the width of this Yui element.
        
        Args:
            value (float): The new width.
        
        Raises:
            TypeError: If value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("width must be a number.")
        self._width = value
        self._needs_graphics_rebuild = True
    @property
    def height(self) -> float:
        """
        Gets the height of this Yui element.
        
        Returns:
            float: The height of the element.
        """
        return self._height
    @height.setter
    def height(self, value:float):
        """
        Sets the height of this Yui element.
        
        Args:
            value (float): The new height.
        
        Raises:
            TypeError: If value is not a number.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("height must be a number.")
        self._height = value
        self._needs_graphics_rebuild = True
    @property
    def size(self) -> Vector2D:
        """
        Gets the size of this Yui element as a Vector2D.
        
        Returns:
            Vector2D: The size of the element.
        """
        return Vector2D(self._width, self._height)
    @size.setter
    def size(self, value:Vector2D|tuple):
        """
        Sets the size of this Yui element.
        
        Args:
            value (Vector2D|tuple): The new size as a Vector2D or a tuple (width, height).
        
        Raises:
            TypeError: If value is not a Vector2D or a tuple of two numbers.
        """
        if isinstance(value, Vector2D):
            self._width, self._height = value.x, value.y
            if self._uses_graphics:
                self._needs_graphics_rebuild = True
        elif isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (int, float)) for v in value):
            self._width, self._height = value
            if self._uses_graphics:
                self._needs_graphics_rebuild = True
        else:
            raise TypeError("size must be a Vector2D or a tuple of two numbers.")

    def _rebuild_graphics(self):
        """
        Rebuilds the graphics for this Yui element.
        This method should be overridden by subclasses to implement custom graphics rendering.
        """
        if not self._uses_graphics:
            return
        if self._destroyed:
            return
        if not self._needs_graphics_rebuild:
            return
        self._graphics = Graphics(self._width, self._height)
        self._needs_graphics_rebuild = False
    def draw(self, graphics:Graphics):
        """
        Draws this Yui element using the provided Graphics object.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        
        Raises:
            RuntimeError: If this Yui element does not use graphics.
        """
        if self._destroyed:
            return
        
        if self.uses_graphics:
            debug_time_start = time.time()
            self._rebuild_graphics()
            self.on_draw(self._graphics)

            self._draw_time_self = time.time() - debug_time_start

            for child in self._children:
                child.draw(graphics)

            with graphics.push_matrix():
                graphics.transform(self.local_matrix)
                graphics.draw(self._graphics)
            
            self._draw_time_subtree = time.time() - debug_time_start
        else:
            debug_time_start = time.time()
            with graphics.push_matrix():
                graphics.transform(self.local_matrix)
                self.on_draw(graphics)
            
                self._draw_time_self = time.time() - debug_time_start

                for child in self._children:
                    child.draw(graphics)
                
            self._draw_time_subtree = time.time() - debug_time_start

    # --- Debug ---
    def print_tree(self):
        """
        Prints the hierarchy of this Yui element and its children.
        This is useful for debugging the structure of the Yui elements.
        """
        indent = ' ' * (self.depth * 2)
        print(f"{indent}{self.__class__.__name__} (x={self.x}, y={self.y}, r={self.r}, sx={self.sx}, sy={self.sy})")
        for child in self.children:
            child.print_tree()
    def draw_bounds(self, graphics:Graphics):
        """
        Draws the bounds of this Yui element for debugging purposes.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        """
        if self._destroyed:
            return
        
        name = self.__class__.__name__
        bounds = self.world_bounds
        # Hue based on depth
        color = Color.from_hue(self.depth / 10.0) # Normalize depth to a hue value

        graphics.fill_color = Color(0, 0, 0, 0) # Transparent fill for bounds
        graphics.stroke_color = color # Stroke color based on depth
        graphics.rect_mode = 'corners' # Center mode for bounds
        graphics.rect(bounds[0], bounds[1], bounds[2], bounds[3])

        graphics.fill_color = color
        graphics.text_size = 12
        graphics.text_align = 0, 0 # Align top left
        graphics.text(name, bounds[0] + 2, bounds[1] + 2) # Draw name at top left of bounds
        

    # --- Flags ---
    @property
    def is_destroyed(self) -> bool:
        """
        Checks if this Yui element has been destroyed.
        
        Returns:
            bool: True if the element is destroyed, False otherwise.
        """
        return self._destroyed
    def destroy(self):
        for child in self.children:
            child.destroy()
        self.on_destroyed()
        if self.parent is not None:
            self.parent.on_child_destroyed(self, self.parent._children.index(self))
            self.parent._children.remove(self)
        self._parent = None

    # --- Callbacks ---
    def on_transform_changed(self, last_x:float, last_y:float, last_r:float, last_sx:float, last_sy:float, last_ax:float, last_ay) -> None:
        """
        Callback for when the transformation of this Yui element changes.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def on_matrix_updated(self, local_matrix:Matrix2D, world_matrix:Matrix2D) -> None:
        """
        Callback for when the transformation matrices are updated.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def on_parent_set(self, parent:Yui) -> None:
        """
        Callback for when the parent of this Yui element is set.
        This can be overridden by subclasses to perform custom actions.
        
        Args:
            parent (Yui): The new parent element.
        """
        pass
    def on_parent_removed(self, old_parent:Yui) -> None:
        """
        Callback for when the parent of this Yui element is removed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            old_parent (Yui): The old parent element that was removed.
        """
        pass
    def on_child_added(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is added.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was added.
            index (int): The index at which the child was added.
        """
        pass
    def on_child_removed(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is removed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was removed.
            index (int): The index at which the child was removed.
        """
        pass
    def on_child_moved(self, child:Yui, old_index:int, new_index:int) -> None:
        """
        Callback for when a child Yui element is moved.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was moved.
            old_index (int): The previous index of the child.
            new_index (int): The new index of the child.
        """
        pass
    def on_child_destroyed(self, child:Yui, index:int) -> None:
        """
        Callback for when a child Yui element is destroyed.
        This can be overridden by subclasses to perform custom actions.

        Args:
            child (Yui): The child element that was destroyed.
            index (int): The index of the child that was destroyed.
        """
        pass
    def on_destroyed(self) -> None:
        """
        Callback for when this Yui element is destroyed.
        This can be overridden by subclasses to perform custom actions.
        """
        pass
    def on_draw(self, graphics:Graphics) -> None:
        """
        Callback for when this Yui element is drawn.
        This can be overridden by subclasses to perform custom drawing actions.
        
        Args:
            graphics (Graphics): The Graphics object to draw on.
        """
        pass
    
    
class YuiRoot(Yui):
    def __init__(self):
        # TODO: Implement root-specific initialization logic.
        """
        Initializes a root Yui element.
        The root element does not have a parent and is the top of the hierarchy.
        """
        super().__init__(parent=None)
    
    
    
        
    

class Mouse():
    pass

class MouseEvent():
    pass

class MouseListener():
    pass

class Keyboard():
    pass

class KeyboardEvent():
    pass

class KeyboardListener():
    pass

root = YuiRoot()
yui = Yui(root)
root.print_tree()
print(yui.parent, yui.root)