#!/usr/bin/env python3
"""
Calibrate GELLO Franka leader arm: home pose and joint signs.

Workflow:
1. Load config and connect to Dynamixel servos
2. Display current joint angles in degrees
3. User manually positions robot to home/upright pose
4. Record home pose as baseline
5. Interactively verify signs by moving each joint one at a time
6. Output calibration results (home pose offsets, joint signs)

Usage:
    python scripts/calibrate_gello_franka_signs_and_home.py --config configs/franka_gello_franka_sim.yaml
"""

import argparse
import time
import yaml
from pathlib import Path
from typing import cast
import numpy as np

# Add gello module to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gello.dynamixel.driver import DynamixelDriver


def load_config(config_path: str) -> dict:
    """Load YAML config."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def degrees_to_radians(deg: float) -> float:
    """Convert degrees to radians."""
    return deg * np.pi / 180.0


def radians_to_degrees(rad: float) -> float:
    """Convert radians to degrees."""
    return rad * 180.0 / np.pi


def continuous_read_loop(driver: DynamixelDriver, duration_seconds: float = 2.0) -> None:
    """Display joint angles for a given duration."""
    print(f"\nReading joint angles for {duration_seconds} seconds (updates every 0.1s)...")
    start = time.time()
    while time.time() - start < duration_seconds:
        pos, vel = driver.get_positions_and_velocities()
        print(f"\r  Joint angles (deg): {[f'{radians_to_degrees(p):7.2f}' for p in pos]}", end="", flush=True)
        time.sleep(0.1)
    print()


def record_home_pose(driver: DynamixelDriver, num_joints: int) -> np.ndarray:
    """Record current position as home pose."""
    print("\n" + "=" * 70)
    print("RECORD HOME POSE")
    print("=" * 70)
    print("Position the robot in the desired home/upright pose.")
    print("Press ENTER when ready to record...")
    input()
    
    pos, _ = driver.get_positions_and_velocities()
    home_pose = pos[:num_joints]
    
    print("\nHome pose recorded (radians):")
    print(f"  {home_pose}")
    print("\nHome pose (degrees):")
    home_deg = np.array([radians_to_degrees(p) for p in home_pose])
    print(f"  {home_deg}")
    
    return home_pose


def verify_joint_signs(driver: DynamixelDriver, num_joints: int, home_pose: np.ndarray) -> np.ndarray:
    """
    Interactively verify sign of each joint.
    
    For each joint, read current pos, then ask user to move it slightly in positive direction,
    verify that reading increases, confirm or flip sign.
    """
    print("\n" + "=" * 70)
    print("VERIFY JOINT SIGNS")
    print("=" * 70)
    print("For each joint, we will:")
    print("  1. Record the current reading")
    print("  2. Ask you to move the joint in the positive/forward direction")
    print("  3. Record the new reading")
    print("  4. Confirm or flip the sign\n")
    
    joint_signs = np.ones(num_joints, dtype=int)
    
    for i in range(num_joints):
        print(f"\n--- Joint {i} ---")
        print(f"CURRENT HOME POSE: {radians_to_degrees(home_pose[i]):.2f}°")
        
        # Get neutral reading
        pos_before, _ = driver.get_positions_and_velocities()
        deg_before = radians_to_degrees(pos_before[i])
        print(f"Current reading: {deg_before:.2f}°")
        
        # Ask user to move
        print(f"\nPlease move joint {i} slightly in the POSITIVE direction (forward/up).")
        print("Press ENTER when moved...")
        input()
        
        time.sleep(0.5)  # let servo settle
        pos_after, _ = driver.get_positions_and_velocities()
        deg_after = radians_to_degrees(pos_after[i])
        print(f"New reading: {deg_after:.2f}°")
        delta = deg_after - deg_before
        print(f"Δ = {delta:.2f}°")
        
        if delta > 0:
            print("✓ Sign is CORRECT (positive movement → positive reading)")
            joint_signs[i] = 1
        else:
            print("✗ Sign needs to be FLIPPED (positive movement → negative reading)")
            print("  Setting sign = -1")
            joint_signs[i] = -1
        
        # Move back to home
        print(f"\nMoving joint {i} back to home pose ({radians_to_degrees(home_pose[i]):.2f}°)...")
        print("Press ENTER when back at home pose...")
        input()
    
    return joint_signs


def output_calibration(
    home_pose: np.ndarray,
    joint_signs: np.ndarray,
    config_path: str,
    output_file: str = "calibration_results.txt"
) -> None:
    """Output calibration results to file and terminal."""
    
    home_deg = np.array([radians_to_degrees(p) for p in home_pose])
    
    results = f"""
GELLO FRANKA CALIBRATION RESULTS
{'='*70}

CONFIG FILE: {config_path}
TIMESTAMP: {time.strftime('%Y-%m-%d %H:%M:%S')}

HOME POSE (BASELINE):
  Radians: {home_pose}
  Degrees: {home_deg}

JOINT SIGNS (to apply to calibration):
  {joint_signs}

CALIBRATION OFFSET (to apply in config):
  These are the home pose readings (raw encoder values at home):
  Radians: {home_pose}
  Degrees: {home_deg}

HOW TO USE:
1. Update your config file with:
   - dynamixel.joint_offsets: {list(home_pose)}
   - dynamixel.joint_signs: {list(joint_signs)}

2. These will be applied in the driver as:
   leader_joint = (raw_encoder - offset) * sign

3. After applying, the home pose should read as all zeros.

{'='*70}
"""
    
    print(results)
    
    with open(output_file, 'w') as f:
        f.write(results)
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate GELLO Franka home pose and joint signs")
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    parser.add_argument('--output', default='calibration_results.txt', help='Output file for results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    print(f"Config loaded from: {args.config}")
    print(f"  Servo types: {config['dynamixel']['servo_types']}")
    print(f"  Joint IDs: {config['dynamixel']['joint_ids']}")
    print(f"  Baudrate: {config['dynamixel']['baudrate']}")
    
    # Extract driver config
    baudrate = config['dynamixel'].get('baudrate', 57600)
    joint_ids = config['dynamixel'].get('joint_ids', list(range(8)))
    port = config['dynamixel'].get('port', '/dev/ttyUSB0')
    num_arm_joints = config['dynamixel'].get('num_arm_joints', 7)
    
    print(f"  Arm joints: {num_arm_joints}")
    print(f"  Port: {port}")
    
    # Connect to Dynamixel
    print("\nConnecting to Dynamixel servos...")
    try:
        driver = DynamixelDriver(
            ids=joint_ids,
            servo_types=config['dynamixel']['servo_types'],
            port=port,
            baudrate=baudrate,
        )
        print("✓ Connected successfully")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        return
    
    try:
        # Read initial pose
        print("\n" + "=" * 70)
        print("INITIAL POSE READING")
        print("=" * 70)
        continuous_read_loop(driver, duration_seconds=2.0)
        
        # Record home pose
        home_pose = record_home_pose(driver, num_arm_joints)
        
        # Verify signs
        joint_signs = verify_joint_signs(driver, num_arm_joints, home_pose)
        
        # Output results
        output_calibration(home_pose, joint_signs, args.config, args.output)
        
    finally:
        print("\nClosing driver...")
        driver.close()
        print("Done.")


if __name__ == '__main__':
    main()
