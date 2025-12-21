#!/bin/bash

# Setup script for ManiSkill PickCube training with NaiviBridger
# This script processes the ManiSkill demonstrations and sets up the dataset

echo "================================================"
echo "ManiSkill PickCube -> NaiviBridger Setup Script"
echo "================================================"

# Configuration
DEMO_PATH="${HOME}/.maniskill/demos/PickCube-v1/motionplanning"
H5_FILE="trajectory.rgb.pd_ee_delta_pos.physx_cpu.h5"  # 使用存在的文件
OUTPUT_DIR="../dataset/pickcube_maniskill"
NUM_TRAJS=100
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Step 1: Check if demo path exists
echo ""
echo "Step 1: Checking ManiSkill demos..."
if [ ! -d "$DEMO_PATH" ]; then
    echo "❌ Error: ManiSkill demo directory not found at $DEMO_PATH"
    echo "Please ensure ManiSkill is installed and demonstrations are generated."
    echo ""
    echo "To generate demonstrations, run:"
    echo "  python -m mani_skill.trajectory.replay_trajectory \\"
    echo "    --traj-path ~/.maniskill/demos/PickCube-v1/motionplanning/trajectory.h5 \\"
    echo "    --use-first-env-state -c pd_ee_delta_pos -o rgbd \\"
    echo "    --save-traj --num-procs 10"
    exit 1
fi

echo "✅ Found ManiSkill demos at $DEMO_PATH"

# List available h5 files
echo ""
echo "Available demonstration files:"
ls -lh "$DEMO_PATH"/*.h5

# Step 2: Process demonstrations
echo ""
echo "Step 2: Processing ManiSkill demonstrations..."
echo "  Source: $DEMO_PATH/$H5_FILE"
echo "  Output: $OUTPUT_DIR"
echo "  Number of trajectories: $NUM_TRAJS"
echo ""

python "$SCRIPT_DIR/process_maniskill_pickcube.py" \
    --demo-path "$DEMO_PATH" \
    --h5-file "$H5_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --num-trajs $NUM_TRAJS

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to process demonstrations"
    exit 1
fi

# Step 3: Verify dataset
echo ""
echo "Step 3: Verifying dataset..."
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ Error: Output directory not created"
    exit 1
fi

TRAJ_COUNT=$(ls -d "$OUTPUT_DIR"/pickcube_traj* 2>/dev/null | wc -l)
echo "✅ Created $TRAJ_COUNT trajectory folders"

if [ ! -f "$OUTPUT_DIR/traj_names.txt" ]; then
    echo "❌ Error: traj_names.txt not found"
    exit 1
fi

echo "✅ traj_names.txt created"

# Step 4: Show dataset statistics
echo ""
echo "Step 4: Dataset statistics"
echo "------------------------"
echo "Total trajectories: $TRAJ_COUNT"

# Check a sample trajectory
SAMPLE_TRAJ="$OUTPUT_DIR/pickcube_traj0"
if [ -d "$SAMPLE_TRAJ" ]; then
    IMG_COUNT=$(ls "$SAMPLE_TRAJ"/*.jpg 2>/dev/null | wc -l)
    echo "Sample trajectory (pickcube_traj0):"
    echo "  - Images: $IMG_COUNT"
    echo "  - Has traj_data.pkl: $([ -f "$SAMPLE_TRAJ/traj_data.pkl" ] && echo "Yes" || echo "No")"
fi

echo ""
echo "================================================"
echo "✅ Setup complete!"
echo "================================================"
echo ""
echo "Next steps:"
echo "1. (Optional) Review the dataset in $OUTPUT_DIR"
echo "2. Start training with:"
echo "   cd maniskill_pickcube && python train_pickcube_maniskill.py --config pickcube_maniskill.yaml"
echo ""
echo "To customize training:"
echo "   - Edit configuration: maniskill_pickcube/pickcube_maniskill.yaml"
echo "   - Adjust dataset parameters in: train/vint_train/data/data_config.yaml"
echo ""
