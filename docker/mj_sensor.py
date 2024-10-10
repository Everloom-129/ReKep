from uitb.simulator import Simulator
from uitb.perception.vision import VisionModule
from uitb.perception.proprioception import ProprioceptionModule

# Create perception modules
vision_module = VisionModule()
proprioception_module = ProprioceptionModule()

# Create simulator with perception modules
simulator = Simulator(
    task_file="humanoid.xml",
    biomechanical_model_file="humanoid.xml",
    perception_modules=[vision_module, proprioception_module]
)

# Start simulation
simulator.reset()

# Run simulation loop
for _ in range(1000):
    action = your_policy(simulator.get_observation())
    simulator.step(action)
    
    # Access perception data
    vision_data = vision_module.get_data()
    proprioception_data = proprioception_module.get_data()
    
    # Use perception data in your logic
    process_perception(vision_data, proprioception_data)