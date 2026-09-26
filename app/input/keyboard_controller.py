"""
Keyboard Input Controller module for MotionVerse.
Encapsulates low-level OS keyboard event dispatch with discrete tap semantics and safety controls.
Strictly decoupled from gesture recognition and game mapping.
"""

import time
import threading
from typing import Optional, Set, List, Callable, Dict, Any


class KeyboardController:
    """
    Dispatches discrete keyboard events to the active OS window.
    
    Features:
    - Completely isolates low-level keyboard libraries (e.g. pyautogui).
    - Asynchronous / non-blocking key tap dispatch to maintain maximum camera FPS.
    - Safety tracker for currently held keys and instant emergency release.
    - Mock mode for automated testing without sending real OS keystrokes.
    - Performance telemetry for dispatch latency and throughput.
    """

    def __init__(
        self,
        key_tap_duration_ms: int = 30,
        mock_mode: bool = False,
        backend_callback: Optional[Callable[[str, str], None]] = None
    ):
        self.key_tap_duration_ms = key_tap_duration_ms
        self.mock_mode = mock_mode
        self.backend_callback = backend_callback

        # Safety & State
        self._active_keys: Set[str] = set()
        self._lock = threading.Lock()
        self._last_dispatch_latency_ms: float = 0.0
        self._last_dispatched_key: Optional[str] = None
        self._last_dispatched_timestamp: float = 0.0
        self._total_dispatches: int = 0

        # Dispatched history (for telemetry & testing)
        self.dispatch_history: List[Dict[str, Any]] = []

        # Initialize pyautogui if not in mock mode
        self._pyautogui = None
        self._backend_error: Optional[str] = None
        if not self.mock_mode:
            try:
                import pyautogui
                # Disable pyautogui default pause to prevent blocking
                pyautogui.PAUSE = 0.0
                pyautogui.FAILSAFE = False
                self._pyautogui = pyautogui
            except ImportError:
                # Do not pretend that real key events are being sent when the
                # optional backend is missing.
                self._backend_error = (
                    "PyAutoGUI is unavailable. Install dependencies with "
                    "`python -m pip install -r requirements.txt`."
                )

    @property
    def is_available(self) -> bool:
        """Whether this controller can emit the configured kind of input."""
        return self.mock_mode or self._pyautogui is not None

    @property
    def backend_error(self) -> Optional[str]:
        """Reason live input cannot be sent, if any."""
        return self._backend_error

    def tap(self, key: str, duration_ms: Optional[int] = None) -> bool:
        """
        Perform a discrete key tap (KEY_DOWN -> wait -> KEY_UP) asynchronously.
        
        Args:
            key: Standard key name (e.g. 'space', 'left', 'right', 'down', 'up').
            duration_ms: Tap duration in milliseconds. Defaults to instance configuration.
            
        Returns:
            True if dispatch initiated successfully.
        """
        if not key or not isinstance(key, str):
            return False
        if not self.is_available:
            return False

        t_start = time.perf_counter()
        dur_ms = duration_ms if duration_ms is not None else self.key_tap_duration_ms
        dur_sec = max(0.005, dur_ms / 1000.0)

        # Standardize key names (e.g. arrow keys)
        normalized_key = key.lower().strip()

        # Update telemetry
        self._last_dispatched_key = normalized_key
        self._last_dispatched_timestamp = time.time()
        self._total_dispatches += 1

        def _perform_tap():
            try:
                with self._lock:
                    self._active_keys.add(normalized_key)

                if self.mock_mode or self._pyautogui is None:
                    if self.backend_callback:
                        self.backend_callback("press", normalized_key)
                else:
                    self._pyautogui.keyDown(normalized_key)
                    time.sleep(dur_sec)
                    self._pyautogui.keyUp(normalized_key)

            except Exception as e:
                pass
            finally:
                with self._lock:
                    self._active_keys.discard(normalized_key)

        # Execute tap in lightweight daemon thread to avoid blocking vision loop
        threading.Thread(target=_perform_tap, daemon=True).start()

        self._last_dispatch_latency_ms = (time.perf_counter() - t_start) * 1000.0

        event_record = {
            "timestamp": self._last_dispatched_timestamp,
            "key": normalized_key,
            "duration_ms": dur_ms,
            "latency_ms": self._last_dispatch_latency_ms
        }
        self.dispatch_history.append(event_record)
        if len(self.dispatch_history) > 100:
            self.dispatch_history.pop(0)

        return True

    def key_down(self, key: str) -> bool:
        """Press and hold a key down."""
        if not key:
            return False
        if not self.is_available:
            return False
        normalized_key = key.lower().strip()
        with self._lock:
            self._active_keys.add(normalized_key)

        if self.mock_mode or self._pyautogui is None:
            if self.backend_callback:
                self.backend_callback("down", normalized_key)
        else:
            try:
                self._pyautogui.keyDown(normalized_key)
            except Exception:
                return False
        return True

    def key_up(self, key: str) -> bool:
        """Release a held key."""
        if not key:
            return False
        if not self.is_available:
            return False
        normalized_key = key.lower().strip()
        with self._lock:
            self._active_keys.discard(normalized_key)

        if self.mock_mode or self._pyautogui is None:
            if self.backend_callback:
                self.backend_callback("up", normalized_key)
        else:
            try:
                self._pyautogui.keyUp(normalized_key)
            except Exception:
                return False
        return True

    def release_all(self) -> None:
        """Emergency release for all currently tracked held keys."""
        with self._lock:
            keys_to_release = list(self._active_keys)
            self._active_keys.clear()

        for k in keys_to_release:
            if not self.mock_mode and self._pyautogui:
                try:
                    self._pyautogui.keyUp(k)
                except Exception:
                    pass
            elif self.backend_callback:
                self.backend_callback("up", k)

    @property
    def dispatch_latency_ms(self) -> float:
        """Latency of initiating the last keyboard event."""
        return self._last_dispatch_latency_ms

    @property
    def last_dispatched_key(self) -> Optional[str]:
        return self._last_dispatched_key

    @property
    def total_dispatches(self) -> int:
        return self._total_dispatches

    @property
    def active_keys(self) -> Set[str]:
        with self._lock:
            return set(self._active_keys)
