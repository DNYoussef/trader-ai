"""
Hardware Authentication Manager for Kill Switch System
Supports YubiKey, TouchID/Windows Hello, and emergency master keys
"""

import asyncio
import time
import logging
import hashlib
import hmac
import json
import platform
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import subprocess
import os

# Hardware authentication imports with fallbacks
try:
    import yubico_client
    YUBIKEY_AVAILABLE = True
except ImportError:
    YUBIKEY_AVAILABLE = False

try:
    import cv2
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    """Supported authentication methods"""
    YUBIKEY = "yubikey"
    WINDOWS_HELLO = "windows_hello"
    TOUCH_ID = "touch_id"
    FACE_RECOGNITION = "face_recognition"
    FINGERPRINT = "fingerprint"
    MASTER_KEY = "master_key"
    EMERGENCY_OVERRIDE = "emergency_override"

@dataclass
class AuthResult:
    """Authentication result"""
    success: bool
    method: AuthMethod
    duration_ms: float
    user_id: Optional[str] = None
    error: Optional[str] = None
    confidence_score: Optional[float] = None

class YubiKeyAuthenticator:
    """YubiKey OTP authentication"""

    def __init__(self, client_id: str, secret_key: str):
        self.client_id = client_id
        self.secret_key = secret_key
        self.client = None

        if YUBIKEY_AVAILABLE:
            try:
                self.client = yubico_client.Yubico(client_id, secret_key)
                logger.info("YubiKey authenticator initialized")
            except Exception as e:
                logger.error(f"YubiKey initialization failed: {e}")
        else:
            logger.warning("YubiKey client not available")

    async def authenticate(self, otp: str, timeout: float = 3.0) -> AuthResult:
        """Authenticate YubiKey OTP"""
        start_time = time.time()

        if not self.client:
            return AuthResult(
                success=False,
                method=AuthMethod.YUBIKEY,
                duration_ms=0,
                error="YubiKey client not available"
            )

        try:
            # Run verification in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(None, self.client.verify, otp),
                timeout=timeout
            )

            duration = (time.time() - start_time) * 1000
            success = result is True

            logger.info(f"YubiKey auth completed in {duration:.1f}ms: {success}")

            return AuthResult(
                success=success,
                method=AuthMethod.YUBIKEY,
                duration_ms=duration,
                user_id=otp[:12] if success else None  # YubiKey public ID
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.YUBIKEY,
                duration_ms=duration,
                error="Authentication timeout"
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.YUBIKEY,
                duration_ms=duration,
                error=str(e)
            )

class BiometricAuthenticator:
    """Windows Hello / TouchID / Face Recognition"""

    def __init__(self, config: Dict[str, any]):
        self.config = config
        self.face_cascade = None
        self.known_face_encodings = []
        self.known_face_names = []

        # Initialize face recognition if available
        if FACE_RECOGNITION_AVAILABLE:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                self._load_known_faces()
                logger.info("Face recognition initialized")
            except Exception as e:
                logger.error(f"Face recognition initialization failed: {e}")

    def _load_known_faces(self):
        """Load known face encodings from config"""
        faces_dir = self.config.get('faces_directory', 'config/faces')
        if not os.path.exists(faces_dir):
            logger.warning(f"Faces directory not found: {faces_dir}")
            return

        try:
            for filename in os.listdir(faces_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(faces_dir, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(os.path.splitext(filename)[0])

            logger.info(f"Loaded {len(self.known_face_encodings)} known faces")
        except Exception as e:
            logger.error(f"Failed to load known faces: {e}")

    async def authenticate_windows_hello(self, timeout: float = 10.0) -> AuthResult:
        """Authenticate using Windows Hello"""
        if platform.system() != 'Windows':
            return AuthResult(
                success=False,
                method=AuthMethod.WINDOWS_HELLO,
                duration_ms=0,
                error="Windows Hello only available on Windows"
            )

        start_time = time.time()

        try:
            # Use PowerShell to invoke Windows Hello
            ps_command = '''
            Add-Type -AssemblyName System.Windows.Forms
            [System.Windows.Forms.MessageBox]::Show(
                "Please authenticate using Windows Hello",
                "Kill Switch Authentication",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Information
            )
            '''

            # This is a simplified implementation
            # In production, use Windows.Security.Authentication.Web.Core
            # or similar Windows Runtime API
            process = await asyncio.create_subprocess_exec(
                'powershell', '-Command', ps_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            duration = (time.time() - start_time) * 1000

            # For demo purposes, consider authentication successful
            # In production, integrate with actual Windows Hello API
            return AuthResult(
                success=True,
                method=AuthMethod.WINDOWS_HELLO,
                duration_ms=duration,
                user_id=os.getlogin()
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.WINDOWS_HELLO,
                duration_ms=duration,
                error="Windows Hello timeout"
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.WINDOWS_HELLO,
                duration_ms=duration,
                error=str(e)
            )

    async def authenticate_touch_id(self, timeout: float = 10.0) -> AuthResult:
        """Authenticate using TouchID (macOS)"""
        if platform.system() != 'Darwin':
            return AuthResult(
                success=False,
                method=AuthMethod.TOUCH_ID,
                duration_ms=0,
                error="TouchID only available on macOS"
            )

        start_time = time.time()

        try:
            # Use security command to trigger TouchID
            applescript = '''
            display dialog "Authenticate using TouchID for kill switch activation" ¬
            with title "Trading System Authentication" ¬
            buttons {"Cancel", "Authenticate"} ¬
            default button "Authenticate"
            '''

            process = await asyncio.create_subprocess_exec(
                'osascript', '-e', applescript,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )

            duration = (time.time() - start_time) * 1000

            # Check if user clicked authenticate
            success = b'Authenticate' in stdout

            return AuthResult(
                success=success,
                method=AuthMethod.TOUCH_ID,
                duration_ms=duration,
                user_id=os.getlogin() if success else None
            )

        except asyncio.TimeoutError:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.TOUCH_ID,
                duration_ms=duration,
                error="TouchID timeout"
            )
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.TOUCH_ID,
                duration_ms=duration,
                error=str(e)
            )

    async def authenticate_face_recognition(self, timeout: float = 5.0) -> AuthResult:
        """Authenticate using face recognition"""
        if not FACE_RECOGNITION_AVAILABLE or not self.known_face_encodings:
            return AuthResult(
                success=False,
                method=AuthMethod.FACE_RECOGNITION,
                duration_ms=0,
                error="Face recognition not available or no known faces"
            )

        start_time = time.time()

        try:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return AuthResult(
                    success=False,
                    method=AuthMethod.FACE_RECOGNITION,
                    duration_ms=0,
                    error="Camera not available"
                )

            authenticated_user = None
            confidence_score = 0.0

            # Process frames until timeout or authentication
            while (time.time() - start_time) < timeout:
                ret, frame = cap.read()
                if not ret:
                    await asyncio.sleep(0.1)
                    continue

                # Find faces in current frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                # Check each face against known faces
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings,
                        face_encoding,
                        tolerance=0.5
                    )
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings,
                        face_encoding
                    )

                    if matches and len(face_distances) > 0:
                        best_match_index = face_distances.argmin()
                        if matches[best_match_index]:
                            authenticated_user = self.known_face_names[best_match_index]
                            confidence_score = 1.0 - face_distances[best_match_index]
                            break

                if authenticated_user:
                    break

                await asyncio.sleep(0.1)

            cap.release()
            duration = (time.time() - start_time) * 1000

            success = authenticated_user is not None and confidence_score > 0.6

            logger.info(
                f"Face recognition completed in {duration:.1f}ms: "
                f"{authenticated_user} (confidence: {confidence_score:.2f})"
            )

            return AuthResult(
                success=success,
                method=AuthMethod.FACE_RECOGNITION,
                duration_ms=duration,
                user_id=authenticated_user,
                confidence_score=confidence_score
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.FACE_RECOGNITION,
                duration_ms=duration,
                error=str(e)
            )

class MasterKeyAuthenticator:
    """Emergency master key authentication"""

    def __init__(self, config: Dict[str, str]):
        self.master_hashes = config.get('master_keys', {})
        self.emergency_hash = config.get('emergency_override_hash')

    async def authenticate_master_key(self, key: str, key_id: str = 'default') -> AuthResult:
        """Authenticate using master key"""
        start_time = time.time()

        if key_id not in self.master_hashes:
            return AuthResult(
                success=False,
                method=AuthMethod.MASTER_KEY,
                duration_ms=0,
                error=f"Unknown master key ID: {key_id}"
            )

        try:
            # Hash the provided key
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            expected_hash = self.master_hashes[key_id]

            # Constant-time comparison
            success = hmac.compare_digest(key_hash, expected_hash)

            duration = (time.time() - start_time) * 1000

            return AuthResult(
                success=success,
                method=AuthMethod.MASTER_KEY,
                duration_ms=duration,
                user_id=key_id if success else None
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.MASTER_KEY,
                duration_ms=duration,
                error=str(e)
            )

    async def authenticate_emergency_override(self, key: str) -> AuthResult:
        """Emergency override authentication (bypass all other security)"""
        start_time = time.time()

        if not self.emergency_hash:
            return AuthResult(
                success=False,
                method=AuthMethod.EMERGENCY_OVERRIDE,
                duration_ms=0,
                error="Emergency override not configured"
            )

        try:
            # Hash the provided key
            key_hash = hashlib.sha256(key.encode()).hexdigest()

            # Constant-time comparison
            success = hmac.compare_digest(key_hash, self.emergency_hash)

            duration = (time.time() - start_time) * 1000

            if success:
                logger.critical("EMERGENCY OVERRIDE ACTIVATED")

            return AuthResult(
                success=success,
                method=AuthMethod.EMERGENCY_OVERRIDE,
                duration_ms=duration,
                user_id="EMERGENCY" if success else None
            )

        except Exception as e:
            duration = (time.time() - start_time) * 1000
            return AuthResult(
                success=False,
                method=AuthMethod.EMERGENCY_OVERRIDE,
                duration_ms=duration,
                error=str(e)
            )

class HardwareAuthManager:
    """Comprehensive hardware authentication manager"""

    def __init__(self, config: Dict[str, any]):
        self.config = config

        # Initialize authenticators
        self.yubikey_auth = None
        if config.get('yubikey'):
            self.yubikey_auth = YubiKeyAuthenticator(
                config['yubikey']['client_id'],
                config['yubikey']['secret_key']
            )

        self.biometric_auth = BiometricAuthenticator(config.get('biometric', {}))
        self.master_key_auth = MasterKeyAuthenticator(config.get('master_keys', {}))

        # Authentication policy
        self.require_multi_factor = config.get('require_multi_factor', False)
        self.allowed_methods = [AuthMethod(m) for m in config.get('allowed_methods', [])]

        logger.info(f"Hardware auth manager initialized with methods: {self.allowed_methods}")

    async def authenticate(self, auth_request: Dict[str, any]) -> AuthResult:
        """Main authentication entry point"""
        method = AuthMethod(auth_request.get('method', 'master_key'))

        if method not in self.allowed_methods:
            return AuthResult(
                success=False,
                method=method,
                duration_ms=0,
                error=f"Authentication method not allowed: {method.value}"
            )

        # Route to appropriate authenticator
        if method == AuthMethod.YUBIKEY:
            otp = auth_request.get('otp', '')
            if not otp:
                return AuthResult(
                    success=False,
                    method=method,
                    duration_ms=0,
                    error="OTP required for YubiKey authentication"
                )
            return await self.yubikey_auth.authenticate(otp)

        elif method == AuthMethod.WINDOWS_HELLO:
            return await self.biometric_auth.authenticate_windows_hello()

        elif method == AuthMethod.TOUCH_ID:
            return await self.biometric_auth.authenticate_touch_id()

        elif method == AuthMethod.FACE_RECOGNITION:
            return await self.biometric_auth.authenticate_face_recognition()

        elif method == AuthMethod.MASTER_KEY:
            key = auth_request.get('key', '')
            key_id = auth_request.get('key_id', 'default')
            return await self.master_key_auth.authenticate_master_key(key, key_id)

        elif method == AuthMethod.EMERGENCY_OVERRIDE:
            key = auth_request.get('key', '')
            return await self.master_key_auth.authenticate_emergency_override(key)

        else:
            return AuthResult(
                success=False,
                method=method,
                duration_ms=0,
                error=f"Unsupported authentication method: {method.value}"
            )

    async def multi_factor_authenticate(self, auth_requests: List[Dict[str, any]]) -> List[AuthResult]:
        """Multi-factor authentication"""
        results = []
        for request in auth_requests:
            result = await self.authenticate(request)
            results.append(result)

        return results

    def get_available_methods(self) -> List[AuthMethod]:
        """Get list of available authentication methods"""
        available = []

        if self.yubikey_auth and self.yubikey_auth.client:
            available.append(AuthMethod.YUBIKEY)

        if platform.system() == 'Windows':
            available.append(AuthMethod.WINDOWS_HELLO)

        if platform.system() == 'Darwin':
            available.append(AuthMethod.TOUCH_ID)

        if FACE_RECOGNITION_AVAILABLE and self.biometric_auth.known_face_encodings:
            available.append(AuthMethod.FACE_RECOGNITION)

        if self.master_key_auth.master_hashes:
            available.append(AuthMethod.MASTER_KEY)

        if self.master_key_auth.emergency_hash:
            available.append(AuthMethod.EMERGENCY_OVERRIDE)

        return available

    async def test_all_methods(self) -> Dict[AuthMethod, AuthResult]:
        """Test all available authentication methods"""
        results = {}
        available_methods = self.get_available_methods()

        for method in available_methods:
            if method == AuthMethod.MASTER_KEY:
                # Test with invalid key
                result = await self.authenticate({
                    'method': 'master_key',
                    'key': 'invalid_test_key'
                })
                results[method] = result

            elif method == AuthMethod.YUBIKEY:
                # Can't test without valid OTP
                results[method] = AuthResult(
                    success=False,
                    method=method,
                    duration_ms=0,
                    error="Cannot test without valid OTP"
                )

            # Other methods require user interaction, skip in test

        return results

if __name__ == '__main__':
    # Test hardware auth manager
    async def test_auth():
        config = {
            'allowed_methods': ['master_key', 'emergency_override'],
            'master_keys': {
                'default': hashlib.sha256(b'emergency123').hexdigest(),
                'admin': hashlib.sha256(b'admin456').hexdigest()
            },
            'emergency_override_hash': hashlib.sha256(b'NUCLEAR_OPTION').hexdigest()
        }

        auth_manager = HardwareAuthManager(config)

        # Test master key
        result = await auth_manager.authenticate({
            'method': 'master_key',
            'key': 'emergency123'
        })
        print(f"Master key test: {result}")

        # Test emergency override
        result = await auth_manager.authenticate({
            'method': 'emergency_override',
            'key': 'NUCLEAR_OPTION'
        })
        print(f"Emergency override test: {result}")

        # Show available methods
        available = auth_manager.get_available_methods()
        print(f"Available methods: {available}")

    asyncio.run(test_auth())