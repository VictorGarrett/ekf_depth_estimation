#!/usr/bin/env python3

class PIController:
    def __init__(self, kp, ki, dt, V_nom):
        """
        Simple PI Controller with Saturation and Anti-Windup.
        
        :param kp: Proportional Gain
        :param ki: Integral Gain
        :param dt: Sampling time (seconds)
        :param V_nom: Nominal Voltage (Saturation limit)
        """
        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.V_nom = V_nom
        
        self.integral = 0.0
        self.error_previous = 0.0

    def compute_command(self, target_vel, current_vel):
        """
        Calculates the control output (Voltage).
        """
        error = target_vel - current_vel
        
        # Proportional term
        P_out = self.kp * error
        
        # Integral term (Discrete integration: area of the rectangle)
        self.integral += error * self.dt
        I_out = self.ki * self.integral
        
        # Total output
        output = P_out + I_out
        
        # --- Saturation & Anti-Windup ---
        # If the output exceeds V_nom, we clamp it and 
        # stop the integral from growing further (windup prevention).
        if output > self.V_nom:
            output = self.V_nom
            self.integral -= error * self.dt  # Undo the integration step
        elif output < -self.V_nom:
            output = -self.V_nom
            self.integral -= error * self.dt  # Undo the integration step
            
        return float(output)

    def reset(self):
        """Resets the integral term to zero."""
        self.integral = 0.0
        self.error_previous = 0.0