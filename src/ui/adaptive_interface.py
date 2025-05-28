"""
Adaptive IDE Interface module that adjusts the UI based on detected cognitive states.
"""

import logging
import time
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CognitiveState(Enum):
    """Enum representing different cognitive states."""
    NEUTRAL = "neutral"
    FATIGUE = "fatigue"
    FRUSTRATION = "frustration"

class UIAdaptationManager:
    """
    Manager for adapting the IDE UI based on detected cognitive states.
    This class serves as an interface between the cognitive state detection
    and the actual IDE interface modifications.
    """
    
    def __init__(self, update_interval=10, enable_break_recommendations=True, fatigue_threshold=0.4, sustained_fatigue_period=60):
        """
        Initialize the UI adaptation manager.
        
        Parameters:
        -----------
        update_interval : int
            Interval (in seconds) between UI updates
        enable_break_recommendations : bool
            Whether to enable break recommendations for sustained fatigue
        fatigue_threshold : float
            PERCLOS threshold for fatigue detection (0.0-1.0)
        sustained_fatigue_period : int
            Period (in seconds) of sustained fatigue before recommending a break
        """
        self.current_state = CognitiveState.NEUTRAL
        self.previous_state = None
        self.state_duration = 0
        self.last_update_time = time.time()
        self.update_interval = update_interval
        self.fatigue_count = 0
        self.frustration_count = 0
        self.ui_elements = {}
        
        # Break recommendation settings
        self.enable_break_recommendations = enable_break_recommendations
        self.fatigue_threshold = fatigue_threshold
        self.sustained_fatigue_period = sustained_fatigue_period
        self.last_break_recommendation_time = 0
        self.break_recommendation_cooldown = 300  # 5 minutes cooldown between recommendations
        self.current_fatigue_duration = 0
        self.current_perclos = 0.0
        
        logger.info("UI Adaptation Manager initialized with break recommendations")
    
    def register_ui_element(self, element_id, element):
        """
        Register a UI element to be controlled by the adaptation manager.
        
        Parameters:
        -----------
        element_id : str
            Identifier for the UI element
        element : object
            UI element object with update method
        """
        self.ui_elements[element_id] = element
        logger.info(f"Registered UI element: {element_id}")
    
    def update_cognitive_state(self, state_probabilities, eye_metrics=None):
        """
        Update the current cognitive state based on detection probabilities.
        
        Parameters:
        -----------
        state_probabilities : dict
            Dictionary of cognitive state probabilities
            {'neutral': float, 'fatigue': float, 'frustration': float}
        eye_metrics : dict, optional
            Dictionary of eye metrics including PERCLOS
            {'perclos': float, 'blink_rate': float, ...}
        """
        # Find the state with the highest probability
        max_prob = -1
        max_state = "neutral"
        
        for state, prob in state_probabilities.items():
            if prob > max_prob:
                max_prob = prob
                max_state = state
        
        if max_state == "neutral":
            new_state = CognitiveState.NEUTRAL
        elif max_state == "fatigue":
            new_state = CognitiveState.FATIGUE
        elif max_state == "frustration":
            new_state = CognitiveState.FRUSTRATION
        else:
            new_state = CognitiveState.NEUTRAL
        
        # Track PERCLOS for break recommendations
        if eye_metrics and 'perclos' in eye_metrics:
            self.current_perclos = eye_metrics['perclos']
            
            # Check if PERCLOS exceeds fatigue threshold
            if self.current_perclos > self.fatigue_threshold:
                self.current_fatigue_duration += self.update_interval
                
                # Check if sustained fatigue period is reached and break recommendation is enabled
                current_time = time.time()
                if (self.enable_break_recommendations and 
                    self.current_fatigue_duration >= self.sustained_fatigue_period and 
                    current_time - self.last_break_recommendation_time > self.break_recommendation_cooldown):
                    
                    self._recommend_break()
                    self.last_break_recommendation_time = current_time
                    self.current_fatigue_duration = 0  # Reset duration after recommending a break
            else:
                # Reset fatigue duration if PERCLOS drops below threshold
                self.current_fatigue_duration = 0
            
        # If state changed, reset duration
        if new_state != self.current_state:
            logger.info(f"Cognitive state changed from {self.current_state.value} to {new_state.value}")
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_duration = 0
        else:
            # Increment duration counter
            self.state_duration += 1
            
        # Track fatigue and frustration occurrences
        if new_state == CognitiveState.FATIGUE:
            self.fatigue_count += 1
        elif new_state == CognitiveState.FRUSTRATION:
            self.frustration_count += 1
        
        # Check if it's time to update the UI
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.update_ui()
            self.last_update_time = current_time
    
    def update_ui(self):
        """
        Update the UI based on the current cognitive state.
        """
        logger.info(f"Updating UI for cognitive state: {self.current_state.value}")
        
        # Apply UI changes for each registered element
        for element_id, element in self.ui_elements.items():
            element.update(self.current_state)
        
        # Apply specific UI adaptations based on state
        if self.current_state == CognitiveState.NEUTRAL:
            self._apply_neutral_state_ui()
        elif self.current_state == CognitiveState.FATIGUE:
            self._apply_fatigue_state_ui()
        elif self.current_state == CognitiveState.FRUSTRATION:
            self._apply_frustration_state_ui()
    
    def _apply_neutral_state_ui(self):
        """Apply UI changes for neutral cognitive state."""
        logger.info("Applying neutral state UI adaptations")
        # Standard interface layout
        # Normal font size and spacing
        # All features visible
        # Regular notification timing
        self._update_layout_standard()
        self._update_font_size_normal()
        self._update_spacing_normal()
        self._show_all_features()
        self._set_notification_timing_regular()
    
    def _apply_fatigue_state_ui(self):
        """Apply UI changes for fatigue cognitive state."""
        logger.info("Applying fatigue state UI adaptations")
        # Reduced sidebar width
        # Increased font size (14px)
        # Enhanced line spacing
        # Break suggestions appear
        # Non-essential features hidden
        self._update_layout_reduced_sidebar()
        self._update_font_size_increased()
        self._update_spacing_enhanced()
        self._show_break_suggestion()
        self._hide_nonessential_features()
    
    def _apply_frustration_state_ui(self):
        """Apply UI changes for frustration cognitive state."""
        logger.info("Applying frustration state UI adaptations")
        
    def _recommend_break(self):
        """
        Show a break recommendation notification when sustained fatigue is detected.
        This implements the "Adaptive UI Intervention" described in the study protocol.
        """
        notification_message = "Feeling tired? Consider taking a short 5-minute break."
        logger.info(f"Recommending break due to sustained fatigue (PERCLOS: {self.current_perclos:.2f})")
        
        # Display notification through the IDE interface
        for element_id, element in self.ui_elements.items():
            if hasattr(element, 'show_notification'):
                element.show_notification(
                    message=notification_message,
                    notification_type="info",
                    buttons=["Take Break", "Ignore"],
                    duration=10000  # 10 seconds
                )
        # Enhanced error highlighting
        # Expanded help sidebar
        # Contextual assistance
        # Documentation suggestions
        # Problem-solving resources
        self._update_layout_expanded_help()
        self._enhance_error_highlighting()
        self._show_contextual_assistance()
        self._show_documentation_suggestions()
        self._show_problem_solving_resources()
    
    # Helper methods for specific UI adaptations
    
    def _update_layout_standard(self):
        """Update layout to standard configuration."""
        # Implementation would interface with the IDE API
        logger.debug("Standard layout applied")
        
    def _update_layout_reduced_sidebar(self):
        """Update layout to reduce sidebar width."""
        # Implementation would interface with the IDE API
        logger.debug("Reduced sidebar width applied")
        
    def _update_layout_expanded_help(self):
        """Update layout to expand help sidebar."""
        # Implementation would interface with the IDE API
        logger.debug("Expanded help sidebar applied")
        
    def _update_font_size_normal(self):
        """Set font size to normal."""
        # Implementation would interface with the IDE API
        logger.debug("Normal font size applied")
        
    def _update_font_size_increased(self):
        """Increase font size for better readability."""
        # Implementation would interface with the IDE API
        logger.debug("Increased font size applied")
        
    def _update_spacing_normal(self):
        """Set line spacing to normal."""
        # Implementation would interface with the IDE API
        logger.debug("Normal spacing applied")
        
    def _update_spacing_enhanced(self):
        """Enhance line spacing for better readability."""
        # Implementation would interface with the IDE API
        logger.debug("Enhanced spacing applied")
        
    def _show_all_features(self):
        """Show all IDE features."""
        # Implementation would interface with the IDE API
        logger.debug("All features visible")
        
    def _hide_nonessential_features(self):
        """Hide non-essential features to reduce cognitive load."""
        # Implementation would interface with the IDE API
        logger.debug("Non-essential features hidden")
        
    def _set_notification_timing_regular(self):
        """Set notification timing to regular intervals."""
        # Implementation would interface with the IDE API
        logger.debug("Regular notification timing set")
        
    def _show_break_suggestion(self):
        """Show break suggestion notification."""
        # Only show break suggestion if in fatigue state for extended period
        if self.state_duration > 5:  # After being in fatigue state for 5 intervals
            # Implementation would interface with the IDE API
            hours_coding = 2  # In a real implementation, this would be tracked
            logger.info(f"Showing break suggestion after {hours_coding} hours of coding")
            # Break suggestion: You've been coding for {hours_coding} hours. Consider a 10-minute break.
        
    def _enhance_error_highlighting(self):
        """Enhance error highlighting in the editor."""
        # Implementation would interface with the IDE API
        logger.debug("Enhanced error highlighting applied")
        
    def _show_contextual_assistance(self):
        """Show contextual assistance based on current code context."""
        # Implementation would interface with the IDE API
        logger.debug("Contextual assistance shown")
        
    def _show_documentation_suggestions(self):
        """Show documentation suggestions related to current code."""
        # Implementation would interface with the IDE API
        logger.debug("Documentation suggestions shown")
        
    def _show_problem_solving_resources(self):
        """Show problem-solving resources based on current errors/context."""
        # Implementation would interface with the IDE API
        logger.debug("Problem-solving resources shown")
        
    def get_statistics(self):
        """
        Get statistics about cognitive states detected during the session.
        
        Returns:
        --------
        dict
            Dictionary containing statistics about cognitive states
        """
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "state_duration": self.state_duration,
            "fatigue_count": self.fatigue_count,
            "frustration_count": self.frustration_count
        }


class IDEInterface:
    """
    Interface to the IDE providing methods to adapt the UI.
    This is a base class that would be extended for specific IDEs
    like VS Code, PyCharm, etc.
    """
    
    def __init__(self, ide_name="Generic IDE"):
        """
        Initialize the IDE interface.
        
        Parameters:
        -----------
        ide_name : str
            Name of the IDE
        """
        self.ide_name = ide_name
        logger.info(f"IDE Interface initialized for {ide_name}")
        
    def apply_ui_adaptation(self, cognitive_state):
        """
        Apply UI adaptations based on cognitive state.
        
        Parameters:
        -----------
        cognitive_state : CognitiveState
            Current cognitive state
        
        Returns:
        --------
        bool
            True if adaptations were applied successfully, False otherwise
        """
        logger.info(f"Applying UI adaptations for {cognitive_state.value} state in {self.ide_name}")
        # This method would be implemented by subclasses for specific IDEs
        return True
        

class VSCodeInterface(IDEInterface):
    """Interface for Visual Studio Code IDE."""
    
    def __init__(self):
        """Initialize the VS Code interface."""
        super().__init__(ide_name="Visual Studio Code")
        self.extension_settings = {}
        
    def apply_ui_adaptation(self, cognitive_state):
        """
        Apply UI adaptations for VS Code based on cognitive state.
        
        Parameters:
        -----------
        cognitive_state : CognitiveState
            Current cognitive state
            
        Returns:
        --------
        bool
            True if adaptations were applied successfully, False otherwise
        """
        logger.info(f"Applying VS Code adaptations for {cognitive_state.value} state")
        
        if cognitive_state == CognitiveState.NEUTRAL:
            return self._apply_neutral_state_vscode()
        elif cognitive_state == CognitiveState.FATIGUE:
            return self._apply_fatigue_state_vscode()
        elif cognitive_state == CognitiveState.FRUSTRATION:
            return self._apply_frustration_state_vscode()
        else:
            return False
    
    def _apply_neutral_state_vscode(self):
        """Apply neutral state adaptations for VS Code."""
        settings = {
            "editor.fontSize": 12,
            "editor.lineHeight": 0,  # Default
            "workbench.sideBar.width": 300,  # Default
            "workbench.statusBar.visible": True,
            "workbench.activityBar.visible": True,
            "editor.minimap.enabled": True,
            "breadcrumbs.enabled": True,
            "window.zoomLevel": 0
        }
        return self._update_vscode_settings(settings)
    
    def _apply_fatigue_state_vscode(self):
        """Apply fatigue state adaptations for VS Code."""
        settings = {
            "editor.fontSize": 14,
            "editor.lineHeight": 1.8,
            "workbench.sideBar.width": 200,  # Reduced width
            "workbench.statusBar.visible": True,
            "workbench.activityBar.visible": False,  # Hide non-essential UI
            "editor.minimap.enabled": False,  # Hide non-essential UI
            "breadcrumbs.enabled": False,  # Hide non-essential UI
            "window.zoomLevel": 0.5  # Slightly zoomed in
        }
        result = self._update_vscode_settings(settings)
        
        # Show break suggestion notification
        self._show_vscode_notification(
            "Break suggestion: You've been coding for 2 hours. Consider a 10-minute break.",
            "info"
        )
        
        return result
    
    def _apply_frustration_state_vscode(self):
        """Apply frustration state adaptations for VS Code."""
        settings = {
            "editor.fontSize": 12,
            "editor.lineHeight": 1.4,
            "workbench.sideBar.width": 400,  # Expanded help sidebar
            "workbench.statusBar.visible": True,
            "workbench.activityBar.visible": True,
            "editor.minimap.enabled": True,
            "breadcrumbs.enabled": True,
            "window.zoomLevel": 0,
            # Enhanced error highlighting
            "problems.decorations.enabled": True,
            "editor.renderLineHighlight": "all",
            "editor.guides.highlightActiveIndentation": True
        }
        result = self._update_vscode_settings(settings)
        
        # Show help notification
        self._show_vscode_notification(
            "Need help? Similar issues found in documentation. Click for suggestions.",
            "info"
        )
        
        return result
        
    def _update_vscode_settings(self, settings):
        """
        Update VS Code settings.
        
        Parameters:
        -----------
        settings : dict
            Dictionary of settings to update
            
        Returns:
        --------
        bool
            True if settings were updated successfully, False otherwise
        """
        # In a real implementation, this would update VS Code settings.json
        # For simulation, we just update our local copy
        self.extension_settings.update(settings)
        logger.debug(f"Updated VS Code settings: {settings}")
        return True
        
    def _show_vscode_notification(self, message, notification_type="info"):
        """
        Show a notification in VS Code.
        
        Parameters:
        -----------
        message : str
            Notification message
        notification_type : str
            Type of notification (info, warning, error)
            
        Returns:
        --------
        bool
            True if notification was shown successfully, False otherwise
        """
        # In a real implementation, this would use the VS Code API
        logger.info(f"VS Code notification ({notification_type}): {message}")
        return True
        
    def show_notification(self, message, notification_type="info", buttons=None, duration=5000):
        """
        Show a toast notification in VS Code with optional buttons.
        Implementation of the break recommendation notification for the study protocol.
        
        Parameters:
        -----------
        message : str
            Notification message
        notification_type : str
            Type of notification (info, warning, error)
        buttons : list
            List of button labels (e.g., ["Take Break", "Ignore"])
        duration : int
            Duration in milliseconds to show the notification
            
        Returns:
        --------
        bool
            True if notification was shown successfully, False otherwise
        """
        # In a real implementation, this would use the VS Code API to show a toast notification
        # with buttons. For the study protocol, we're simulating the notification described
        # in "Adaptive UI Intervention" section
        
        logger.info(f"VS Code toast notification: {message}")
        if buttons:
            buttons_str = ", ".join(buttons)
            logger.info(f"Notification buttons: {buttons_str}")
        
        # Return True to indicate success
        return True


# Factory function to create IDE interface based on IDE name
def create_ide_interface(ide_name):
    """
    Create an IDE interface based on IDE name.
    
    Parameters:
    -----------
    ide_name : str
        Name of the IDE
        
    Returns:
    --------
    IDEInterface
        Interface for the specified IDE
    """
    if ide_name.lower() == "vscode" or ide_name.lower() == "visual studio code":
        return VSCodeInterface()
    else:
        logger.warning(f"No specific interface available for {ide_name}, using generic interface")
        return IDEInterface(ide_name)
