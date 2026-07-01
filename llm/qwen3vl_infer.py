#!/usr/bin/env python3
"""
Qwen3-VL-2B-Instruct Inference Test Script
Tests the model with image and text inputs for multimodal reasoning
"""

import os
import sys
import torch
import time
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO

# Import Qwen VL utilities
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

class Qwen3VLInference:
    """Qwen3-VL inference wrapper"""
    
    def __init__(self, model_path=None, device="auto", dtype=torch.float16):
        """
        Initialize the model
        
        Args:
            model_path: Path to the model directory (default: ModelScope cache)
            device: Device to run on ("auto", "cuda", "cpu")
            dtype: Data type for model weights (torch.float16 or torch.bfloat16)
        """
        if model_path is None:
            model_path = os.path.expanduser(
                "~/.cache/modelscope/hub/models/Qwen/Qwen3-VL-2B-Instruct"
            )
        
        print(f"Loading model from: {model_path}")
        print(f"Device: {device}, Dtype: {dtype}")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please download it first using: "
                "modelscope download --model Qwen/Qwen3-VL-2B-Instruct"
            )
        
        # Load model - Qwen3VL uses specific class
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device
        )
        
        # Load processor (handles tokenization and image processing)
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        self.device = next(self.model.parameters()).device
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"✓ Model memory: {self.get_model_memory():.2f} GB")
    
    def get_model_memory(self):
        """Get model memory usage in GB"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / 1024**3
        return 0
    
    def inference(self, messages, max_new_tokens=512, temperature=0.7):
        """
        Run inference on the model
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        # Prepare inputs - following official Qwen3-VL format
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info with video metadata support
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            [messages],
            return_video_kwargs=True,
            image_patch_size=16,
            return_video_metadata=True
        )
        
        # Handle video metadata if present
        video_metadatas = None
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            video_metadata=video_metadatas,
            **video_kwargs,
            do_resize=False,
            padding=True,
            return_tensors="pt"
        )
        
        inputs = inputs.to(self.device)
        
        # Generate
        print("Generating response...")
        start_time = time.time()
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True
            )
        
        # Trim input tokens from output
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        elapsed_time = time.time() - start_time
        tokens_generated = len(generated_ids_trimmed[0])
        tokens_per_sec = tokens_generated / elapsed_time
        
        print(f"✓ Generated {tokens_generated} tokens in {elapsed_time:.2f}s ({tokens_per_sec:.2f} tokens/s)")
        
        return output_text

def load_test_image(image_source):
    """
    Load a test image from URL or local path
    
    Args:
        image_source: URL or local file path
        
    Returns:
        PIL Image object
    """
    if image_source.startswith("http://") or image_source.startswith("https://"):
        print(f"Downloading image from: {image_source}")
        response = requests.get(image_source)
        img = Image.open(BytesIO(response.content))
    else:
        print(f"Loading image from: {image_source}")
        img = Image.open(image_source)
    
    print(f"✓ Image loaded: {img.size}, mode: {img.mode}")
    return img

def test_text_only():
    """Test 1: Text-only inference"""
    print("\n" + "="*80)
    print("TEST 1: Text-only Reasoning")
    print("="*80)
    
    inference_engine = Qwen3VLInference()
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain the concept of autonomous driving in simple terms."}
            ]
        }
    ]
    
    response = inference_engine.inference(messages)
    print(f"\n📝 Response:\n{response}\n")
    
    return inference_engine

def test_image_understanding(inference_engine):
    """Test 2: Image understanding"""
    print("\n" + "="*80)
    print("TEST 2: Image Understanding")
    print("="*80)
    
    # Use a sample driving scene image
    #image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image_url = "demo.jpeg"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Describe this image in detail."}
            ]
        }
    ]
    
    response = inference_engine.inference(messages)
    print(f"\n📝 Response:\n{response}\n")

def test_driving_scene_reasoning(inference_engine):
    """Test 3: Autonomous driving scene reasoning"""
    print("\n" + "="*80)
    print("TEST 3: Driving Scene Reasoning")
    print("="*80)
    
    # Use a road scene image
    # image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image_url = "demo.jpeg"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {
                    "type": "text",
                    "text": "You are an autonomous driving assistant. Analyze this scene and provide:\n"
                            "1. What objects do you see?\n"
                            "2. What are the potential risks?\n"
                            "3. What action should the vehicle take?"
                }
            ]
        }
    ]
    
    response = inference_engine.inference(messages, max_new_tokens=512)
    print(f"\n Response:\n{response}\n")

def test_multi_turn_conversation(inference_engine):
    """Test 4: Multi-turn conversation"""
    print("\n" + "="*80)
    print("TEST 4: Multi-turn Conversation")
    print("="*80)
    
    # image_url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    image_url = "demo.jpeg"
    
    # Turn 1
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "What's in this image?"}
            ]
        }
    ]
    
    response1 = inference_engine.inference(messages, max_new_tokens=256)
    print(f"\n Turn 1 Response:\n{response1}\n")
    
    # Turn 2 - follow-up question
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response1}]
    })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": "What should a self-driving car pay attention to in this scene?"}]
    })
    
    response2 = inference_engine.inference(messages, max_new_tokens=256)
    print(f"\n Turn 2 Response:\n{response2}\n")

def test_local_image(inference_engine, image_path):
    """Test 5: Local image inference"""
    print("\n" + "="*80)
    print("TEST 5: Local Image Inference")
    print("="*80)
    
    if not os.path.exists(image_path):
        print(f"  Image not found: {image_path}")
        print("Skipping local image test")
        return
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{os.path.abspath(image_path)}"},
                {"type": "text", "text": "Describe this image and identify any potential hazards for autonomous driving."}
            ]
        }
    ]
    
    response = inference_engine.inference(messages)
    print(f"\n Response:\n{response}\n")

def test_performance_benchmark(inference_engine):
    """Test 6: Performance benchmark"""
    print("\n" + "="*80)
    print("TEST 6: Performance Benchmark")
    print("="*80)
    
    num_runs = 5
    total_time = 0
    total_tokens = 0
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the key components of an autonomous driving system?"}
            ]
        }
    ]
    
    print(f"Running {num_runs} inference iterations...")
    
    for i in range(num_runs):
        start_time = time.time()
        response = inference_engine.inference(messages, max_new_tokens=128)
        elapsed = time.time() - start_time
        
        # Approximate token count (rough estimate)
        tokens = len(response.split()) * 1.3
        total_time += elapsed
        total_tokens += tokens
        
        print(f"  Run {i+1}: {elapsed:.2f}s, ~{tokens:.0f} tokens")
    
    avg_time = total_time / num_runs
    avg_tokens_per_sec = total_tokens / total_time
    
    print(f"\n Benchmark Results:")
    print(f"  Average time per inference: {avg_time:.2f}s")
    print(f"  Average throughput: {avg_tokens_per_sec:.2f} tokens/s")
    print(f"  GPU Memory: {inference_engine.get_model_memory():.2f} GB")

def main():
    """Main test runner"""
    print("="*80)
    print("Qwen3-VL-2B-Instruct Inference Test Suite")
    print("="*80)
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("  CUDA not available, using CPU (will be slow)")
    
    try:
        # Run all tests
        inference_engine = test_text_only()
        test_image_understanding(inference_engine)
        test_driving_scene_reasoning(inference_engine)
        test_multi_turn_conversation(inference_engine)
        
        # Test with local image if provided
        if len(sys.argv) > 1:
            test_local_image(inference_engine, sys.argv[1])
        
        # Performance benchmark
        test_performance_benchmark(inference_engine)
        
        print("\n" + "="*80)
        print(" All tests completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


