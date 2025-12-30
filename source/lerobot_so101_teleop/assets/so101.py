import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

here = os.path.dirname(os.path.abspath(__file__))

SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{here}/usd/SO-ARM101-USD.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "Rotation": -0.2736,
            "Pitch": -0.6109,
            "Elbow": -0.0745,
            "Wrist_Pitch": 1.5148,
            "Wrist_Roll": -1.6034,
            "Jaw": -0.1465,
        },
    ),
    actuators={
        # ROTATION (Gear: 1/191, Torque: 34.4 N-m)
        "rotation": ImplicitActuatorCfg(
            joint_names_expr=["Rotation"],
            effort_limit_sim=30,
            stiffness=55,        
            damping=0.7,         
        ),
        
        # PITCH (Gear: 1/345, Torque: 62.1 N-m - HIGHEST)
        "pitch": ImplicitActuatorCfg(
            joint_names_expr=["Pitch"],
            effort_limit_sim=30,
            stiffness=30,        
            damping=0.8,         
        ),
        
        # ELBOW (Gear: 1/191, Torque: 34.4 N-m)
        "elbow": ImplicitActuatorCfg(
            joint_names_expr=["Elbow"],
            effort_limit_sim=30,
            stiffness=25,        
            damping=0.7,         
        ),
        
        # WRIST PITCH (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_pitch": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Pitch"],
            effort_limit_sim=30,
            stiffness=12,        
            damping=0.5,         
        ),
        
        # WRIST ROLL (Gear: 1/147, Torque: 26.5 N-m)
        "wrist_roll": ImplicitActuatorCfg(
            joint_names_expr=["Wrist_Roll"],
            effort_limit_sim=30,
            stiffness=7,         
            damping=0.5,         
        ),
        
        # GRIPPER (Gear: 1/147, Torque: 26.5 N-m)
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["Jaw"],
            effort_limit_sim=30,
            stiffness=4,         
            damping=0.3,         
        ),
    },
)




