# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import json
import logging
import pickle
import os
import re
from typing import Dict, List, Tuple, Any
from convlab.policy.policy import Policy
from convlab.util.custom_util import set_seed
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# Add import for distillation buffer
from .distillation_buffer import DistillationBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DyBBT(Policy):
    """
    DyBBT: Dynamic Balance via Bandit-inspired Targeting for Dialog Policy with Cognitive Dual-Systems
    Implements the cognitive dual-system architecture with meta-controller based on the ICLR 2026 paper
    """

    def __init__(self, config: Dict, is_train: bool = False, seed: int = 0, dataset_type: str = "multiwoz"):
        super().__init__()
        
        self.config = config
        self.is_train = is_train
        self.seed = seed
        self.dataset_type = dataset_type  # "multiwoz" or "msdialog"

        self.sess = None
        
        # Load slot co-occurrence matrix
        self.slot_cooccurrence_matrix = self._load_slot_cooccurrence_matrix()
        
        # Cognitive state tracking
        self.cognitive_state_counts = {}  # Track visits to cognitive states
        self.current_cognitive_state = None
        
        # System components
        self.system1 = self._init_system1()
        self.system2 = self._init_system2()
        
        # Meta-controller parameters from paper
        self.tau = self.config['dybbt']['tau']  # Exploration parameter
        self.confidence_threshold = self.config['dybbt']['confidence_threshold']  # κ threshold
        
        # Initialize distillation buffer
        distillation_buffer_size = self.config['dybbt'].get('distillation_buffer_size', 10000)
        self.distillation_buffer = DistillationBuffer(max_size=distillation_buffer_size)
        
        logging.info(f'DyBBT initialized with seed {seed} for dataset: {dataset_type}')
        set_seed(seed)

    def _load_slot_cooccurrence_matrix(self) -> Dict:
        """Load the precomputed slot co-occurrence matrix"""
        matrix_path = self.config['dybbt']['slot_cooccurrence_matrix_path']
        
        # Handle dataset-specific matrix paths
        if self.dataset_type == "msdialog":
            # For MSDialog, use MSDialog-specific matrix if available
            if 'msdialog_matrix_path' in self.config['dybbt']:
                matrix_path = self.config['dybbt']['msdialog_matrix_path']
        else:
            # For MultiWoz, use MultiWoz-specific matrix if available
            if 'multiwoz_matrix_path' in self.config['dybbt']:
                matrix_path = self.config['dybbt']['multiwoz_matrix_path']
        
        try:
            # Try to load as pickle file first
            if matrix_path.endswith('.pkl'):
                import pickle
                with open(matrix_path, 'rb') as f:
                    return pickle.load(f)
            # Try to load as JSON file
            elif matrix_path.endswith('.json'):
                import json
                with open(matrix_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            # Default to pickle
            else:
                import pickle
                with open(matrix_path, 'rb') as f:
                    return pickle.load(f)
        except FileNotFoundError:
            logging.warning(f"Slot co-occurrence matrix not found at {matrix_path}")
            return {}
        except Exception as e:
            logging.warning(f"Error loading slot co-occurrence matrix from {matrix_path}: {e}")
            return {}

    def _init_system1(self):
        """Initialize System 1 (fast intuitive system) with base model + LoRA checkpoint"""
        try:
            # Get model path from config
            model_path = self.config['dybbt'].get('system1_model_path', "./models/system1_lora_finetuned")
            
            # Load base model first
            base_model_name = "Qwen/Qwen3-1.7B"  # Default base model
            if 'system1_base_model' in self.config['dybbt']:
                base_model_name = self.config['dybbt']['system1_base_model']
            
            logging.info(f"Loading System 1 base model: {base_model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token
            
            # Load base model
            model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
            
            # Load LoRA checkpoint if available
            if os.path.exists(model_path):
                logging.info(f"Loading System 1 LoRA checkpoint from: {model_path}")
                model = PeftModel.from_pretrained(model, model_path)
            else:
                logging.warning(f"System 1 LoRA checkpoint not found at {model_path}, using base model only")
            
            # Move model to device
            model.to(DEVICE)
            model.eval()
            
            return {
                'model': model,
                'tokenizer': tokenizer
            }
            
        except Exception as e:
            logging.error(f"Error initializing System 1: {e}")
            return None

    def _init_system2(self):
        """Initialize System 2 (slow reasoning system) with base model only"""
        try:
            # Get model path from config
            model_path = self.config['dybbt'].get('system2_model_path', "./models/system2_base")
            
            # Dataset-specific initialization
            if self.dataset_type == "msdialog":
                logging.info("Initializing System 2 for MSDialog dataset with base model")
            else:
                logging.info("Initializing System 2 for MultiWoz dataset with base model")
            
            # Load base model
            base_model_name = "Qwen/Qwen3-1.7B"  # Default base model
            if 'system2_base_model' in self.config['dybbt']:
                base_model_name = self.config['dybbt']['system2_base_model']
            
            # If a specific base model path is provided, use it
            if os.path.exists(model_path):
                logging.info(f"Loading System 2 base model from: {model_path}")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            else:
                logging.info(f"Loading System 2 base model: {base_model_name}")
                tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
            
            tokenizer.pad_token = tokenizer.eos_token
            
            # Move model to device
            model.to(DEVICE)
            model.eval()
            
            return {
                'model': model,
                'tokenizer': tokenizer
            }
            
        except Exception as e:
            logging.error(f"Error initializing System 2: {e}")
            return None

    def compute_cognitive_state(self, state: Dict, turn: int, max_turns: int) -> Tuple[float, float, float]:
        """
        Compute the cognitive state c_t = [d_t, u_t, ρ_t]
        
        Args:
            state: Current dialog state
            turn: Current turn number
            max_turns: Maximum number of turns
            
        Returns:
            Tuple of (dialogue_progress, user_uncertainty, slot_dependency)
        """
        # 1. Dialogue progress: d_t = t / L
        dialogue_progress = turn / max_turns
        
        # 2. User uncertainty: u_t = |S_unknown| / |S_relevant|
        user_uncertainty = self._compute_user_uncertainty(state)
        
        # 3. Slot dependency: ρ_t = max_{u∈U} (1/|F| ∑_{f∈F} M(u, f))
        slot_dependency = self._compute_slot_dependency(state)
        
        return dialogue_progress, user_uncertainty, slot_dependency

    def _compute_dialogue_progress(self, state: Dict) -> float:
        """Compute dialogue progress based on filled slots"""
        total_slots = 0
        filled_slots = 0
        
        # Count total and filled slots
        if self.dataset_type == "msdialog":
            # MSDialog format handling
            if 'belief_state' in state:
                belief_state = state['belief_state']
                if isinstance(belief_state, dict):
                    for slot, value in belief_state.items():
                        total_slots += 1
                        if value not in ["?", None, "", "not mentioned", "unknown"]:
                            filled_slots += 1
            else:
                # Fallback for direct slot-value structure
                for slot, value in state.items():
                    total_slots += 1
                    if value not in ["?", None, "", "not mentioned", "unknown"]:
                        filled_slots += 1
        else:
            # MultiWOZ format
            if 'belief_state' in state:
                belief_state = state['belief_state']
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            total_slots += 1
                            if value != "?" and value is not None and value != "" and value != "not mentioned":
                                filled_slots += 1
            else:
                # Fallback for direct domain-slot structure
                for domain, slots in state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            total_slots += 1
                            if value != "?" and value is not None and value != "" and value != "not mentioned":
                                filled_slots += 1
        
        if total_slots == 0:
            return 0.0
        
        return filled_slots / total_slots

    def _compute_user_uncertainty(self, state: Dict) -> float:
        """Compute user uncertainty based on unknown slots"""
        unknown_count = 0
        relevant_count = 0
        
        # Handle MSDialog format
        if self.dataset_type == "msdialog":
            # MSDialog uses a different state structure
            if 'belief_state' in state:
                belief_state = state['belief_state']
                if isinstance(belief_state, dict):
                    for slot, value in belief_state.items():
                        # MSDialog uses different unknown indicators
                        if value in ["?"] or value is None or value == "" or value == "not mentioned" or value == "unknown":
                            unknown_count += 1
                        relevant_count += 1
            else:
                # Fallback for direct slot-value structure
                for slot, value in state.items():
                    if value in ["?"] or value is None or value == "" or value == "not mentioned" or value == "unknown":
                        unknown_count += 1
                    relevant_count += 1
        else:
            # MultiWOZ format
            if 'belief_state' in state:
                belief_state = state['belief_state']
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            # Check for unknown/empty values
                            if value == "?" or value is None or value == "" or value == "not mentioned":
                                unknown_count += 1
                            relevant_count += 1
            else:
                # Fallback for direct domain-slot structure
                for domain, slots in state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value == "?" or value is None or value == "" or value == "not mentioned":
                                unknown_count += 1
                            relevant_count += 1
        
        return unknown_count / max(relevant_count, 1)

    def _compute_slot_dependency(self, state: Dict) -> float:
        """Compute slot dependency based on co-occurrence matrix"""
        if not self.slot_cooccurrence_matrix or 'cooccurrence_matrix' not in self.slot_cooccurrence_matrix:
            return 0.0
        
        filled_slots = []
        unknown_slots = []
        
        # Extract filled and unknown slots from state
        if self.dataset_type == "msdialog":
            # MSDialog format handling
            if 'belief_state' in state:
                belief_state = state['belief_state']
                if isinstance(belief_state, dict):
                    for slot, value in belief_state.items():
                        slot_name = slot  # MSDialog uses simple slot names
                        # Check for unknown/empty values
                        if value in ["?"] or value is None or value == "" or value == "not mentioned" or value == "unknown":
                            unknown_slots.append(slot_name)
                        else:
                            filled_slots.append(slot_name)
            else:
                # Fallback for direct slot-value structure
                for slot, value in state.items():
                    slot_name = slot
                    if value in ["?"] or value is None or value == "" or value == "not mentioned" or value == "unknown":
                        unknown_slots.append(slot_name)
                    else:
                        filled_slots.append(slot_name)
        else:
            # MultiWOZ format
            if 'belief_state' in state:
                belief_state = state['belief_state']
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            slot_name = f"{domain}-{slot}"
                            # Check for unknown/empty values
                            if value == "?" or value is None or value == "" or value == "not mentioned":
                                unknown_slots.append(slot_name)
                            else:
                                filled_slots.append(slot_name)
            else:
                # Fallback for direct domain-slot structure
                for domain, slots in state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            slot_name = f"{domain}-{slot}"
                            if value == "?" or value is None or value == "" or value == "not mentioned":
                                unknown_slots.append(slot_name)
                            else:
                                filled_slots.append(slot_name)
        
        if not filled_slots or not unknown_slots:
            return 0.0
        
        # Compute maximum dependency
        max_dependency = 0.0
        cooccurrence_matrix = self.slot_cooccurrence_matrix['cooccurrence_matrix']
        
        for unknown_slot in unknown_slots:
            dependency_sum = 0.0
            filled_count = 0
            
            for filled_slot in filled_slots:
                # Check if both slots exist in the co-occurrence matrix
                if unknown_slot in cooccurrence_matrix and filled_slot in cooccurrence_matrix[unknown_slot]:
                    dependency_sum += cooccurrence_matrix[unknown_slot][filled_slot]
                    filled_count += 1
            
            if filled_count > 0:
                avg_dependency = dependency_sum / filled_count
                max_dependency = max(max_dependency, avg_dependency)
        
        return max_dependency

    def should_activate_system2(self, cognitive_state: Tuple, system1_confidence: float) -> bool:
        """
        Meta-controller decision: whether to activate System 2
        Based on dynamic balance principle: N_t(c_t) < τ * sqrt(log T)
        
        Args:
            cognitive_state: Tuple of (dialogue_progress, user_uncertainty, slot_dependency)
            system1_confidence: Confidence score from System 1 prediction
            
        Returns:
            True if System 2 should be activated, False otherwise
        """
        d_t, u_t, ρ_t = cognitive_state
        
        # Discretize cognitive state to 5 bins as per paper
        num_bins = 5
        d_bin = min(int(d_t * num_bins), num_bins - 1)
        u_bin = min(int(u_t * num_bins), num_bins - 1)
        rho_bin = min(int(ρ_t * num_bins), num_bins - 1)
        
        # Create discrete cognitive state key
        cognitive_state_key = f"({d_bin}, {u_bin}, {rho_bin})"
        
        # Get visit count for this cognitive state
        visit_count = self.cognitive_state_counts.get(cognitive_state_key, 0)
        
        # Dynamic balance criterion from paper: n_t(c_t) < τ * sqrt(log T)
        tau = self.tau
        total_turns = self.config['dybbt']['max_turns']
        
        # Handle edge case where total_turns is too small
        if total_turns <= 1:
            dynamic_criterion = visit_count < tau
        else:
            dynamic_criterion = visit_count < tau * np.sqrt(np.log(total_turns))
        
        # Confidence condition from paper: p_t^{S1} < κ
        low_confidence = system1_confidence < self.confidence_threshold
        
        # Update visit count
        self.cognitive_state_counts[cognitive_state_key] = visit_count + 1
        self.current_cognitive_state = cognitive_state_key
        
        # Log meta-controller decision
        if dynamic_criterion or low_confidence:
            logging.info(f"Meta-controller: Activating System 2. "
                        f"Cognitive state: {cognitive_state_key}, "
                        f"Visit count: {visit_count}, "
                        f"S1 confidence: {system1_confidence:.3f}, "
                        f"Dynamic criterion: {dynamic_criterion}, "
                        f"Low confidence: {low_confidence}")
        
        # Return true if either condition is met (disjunctive design)
        return dynamic_criterion or low_confidence

    def system1_predict(self, state: Dict, cognitive_state: Tuple) -> Tuple[List, float]:
        """
        System 1 prediction (fast intuitive response)
        Uses the LoRA-finetuned Qwen3-1.7B model for actual inference
        """
        try:
            # Check if System 1 model is loaded
            if self.system1 is None:
                logging.warning("System 1 model not loaded, using fallback heuristic")
                return self._system1_fallback_predict(state, cognitive_state)
            
            dialogue_progress, user_uncertainty, slot_dependency = cognitive_state
            
            # Construct prompt for System 1
            belief_state_str = json.dumps(state.get('belief_state', {}), ensure_ascii=False, indent=2)
            
            system1_prompt = f"""
You are the fast intuitive component (System 1) of a dialogue system. Your task is to generate the most appropriate next action based on the current cognitive state.

**Current Cognitive State:**
- Dialogue Progress (d_t): {dialogue_progress:.3f}
- User Uncertainty (u_t): {user_uncertainty:.3f}  
- Slot Dependency (ρ_t): {slot_dependency:.3f}

**Current Belief State:**
{belief_state_str}

**Instructions:**
Generate the most appropriate next system action in the format: [action_type, domain, slot, value]

Available action types: request, inform, confirm, book, general

Output ONLY the action in the specified format, nothing else.
"""
            
            # Tokenize and generate response
            tokenizer = self.system1['tokenizer']
            model = self.system1['model']
            
            inputs = tokenizer(system1_prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract action from response
            action = self._parse_system1_response(response)
            
            # Calculate confidence based on response quality and cognitive state
            confidence = self._calculate_system1_confidence(response, cognitive_state)
            
            return action, confidence
            
        except Exception as e:
            logging.error(f"Error in System 1 prediction: {e}")
            # Fallback to heuristic prediction
            return self._system1_fallback_predict(state, cognitive_state)

    def _parse_system1_response(self, response: str) -> List:
        """Parse the response from System 1 model to extract action"""
        try:
            # Look for action pattern in the response
            import re
            
            # Pattern to match action format: ["action_type", "domain", "slot", "value"]
            action_pattern = r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']\s*\]'
            
            matches = re.findall(action_pattern, response)
            if matches:
                action_type, domain, slot, value = matches[0]
                return [[action_type, domain, slot, value]]
            
            # Fallback: try to extract action components separately
            action_types = ["request", "inform", "confirm", "book", "general"]
            for action_type in action_types:
                if action_type in response.lower():
                    # Try to extract domain and slot from context
                    domain = ""
                    slot = ""
                    value = ""
                    
                    # Simple heuristic for domain and slot extraction
                    if "hotel" in response.lower():
                        domain = "hotel"
                    elif "restaurant" in response.lower():
                        domain = "restaurant"
                    elif "attraction" in response.lower():
                        domain = "attraction"
                    elif "train" in response.lower():
                        domain = "train"
                    elif "taxi" in response.lower():
                        domain = "taxi"
                    
                    return [[action_type, domain, slot, value]]
            
            # Ultimate fallback
            return [["general", "bye", ""]]
            
        except Exception as e:
            logging.warning(f"Error parsing System 1 response: {e}")
            return [["general", "bye", ""]]

    def _construct_system2_prompt(self, state: Dict, cognitive_state: Tuple) -> str:
        """Construct prompt for System 2 reasoning"""
        d_t, u_t, ρ_t = cognitive_state
        
        prompt = f"""You are an AI assistant for task-oriented dialogue systems. Analyze the current dialogue state and generate reasoning paths.

Current State:
- Dialogue Progress: {d_t:.2f}
- User Uncertainty: {u_t:.2f}
- Slot Dependency: {ρ_t:.2f}
- Belief State: {state.get('belief_state', {})}

Generate 3 distinct reasoning paths with:
1. A rationale explaining the strategy
2. A sequence of actions to take
3. A confidence score (0.0-1.0)

Format each path as:
PATH 1:
RATIONALE: [rationale]
ACTIONS: [action1, action2, ...]
CONFIDENCE: [score]

PATH 2:
RATIONALE: [rationale]
ACTIONS: [action1, action2, ...]
CONFIDENCE: [score]

PATH 3:
RATIONALE: [rationale]
ACTIONS: [action1, action2, ...]
CONFIDENCE: [score]

Focus on different strategies based on the cognitive state."""
        
        return prompt

    def _parse_system2_response(self, response: str) -> List[Dict]:
        """Parse System 2 model response to extract reasoning paths"""
        reasoning_paths = []
        
        # Split response by path markers
        path_sections = response.split('PATH ')
        
        for section in path_sections[1:]:  # Skip first empty section
            path_data = {}
            
            # Extract rationale
            rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=\n(?:ACTIONS|PATH|$))', section, re.DOTALL)
            if rationale_match:
                path_data['rationale'] = rationale_match.group(1).strip()
            
            # Extract actions
            actions_match = re.search(r'ACTIONS:\s*(.+?)(?=\n(?:CONFIDENCE|PATH|$))', section, re.DOTALL)
            if actions_match:
                actions_str = actions_match.group(1).strip()
                # Parse action list
                path_data['action_sequence'] = self._parse_action_string(actions_str)
            
            # Extract confidence
            confidence_match = re.search(r'CONFIDENCE:\s*([0-9.]+)', section)
            if confidence_match:
                path_data['confidence'] = float(confidence_match.group(1))
            
            # Only add valid paths
            if all(key in path_data for key in ['rationale', 'action_sequence', 'confidence']):
                path_data['sequence_id'] = len(reasoning_paths) + 1
                reasoning_paths.append(path_data)
        
        return reasoning_paths

    def _parse_action_string(self, actions_str: str) -> List[List[str]]:
        """Parse action string into list of action tuples"""
        actions = []
        
        # Handle different formats
        if '[' in actions_str and ']' in actions_str:
            # Parse as Python list format
            try:
                import ast
                parsed_actions = ast.literal_eval(actions_str)
                if isinstance(parsed_actions, list):
                    actions = parsed_actions
            except:
                pass
        
        # Fallback: parse as comma-separated
        if not actions:
            action_items = actions_str.split(',')
            for item in action_items:
                item = item.strip()
                if item:
                    # Simple action format: [domain, act, slot]
                    parts = item.split()
                    if len(parts) >= 2:
                        domain = parts[0] if parts[0] != 'general' else 'general'
                        act = parts[1]
                        slot = parts[2] if len(parts) > 2 else ''
                        actions.append([domain, act, slot])
        
        return actions

    def _system2_fallback_reason(self, state: Dict, cognitive_state: Tuple) -> List[Dict]:
        """Fallback heuristic reasoning for System 2 when model fails"""
        d_t, u_t, ρ_t = cognitive_state
        
        reasoning_paths = []
        
        # Path 1: Direct strategy - focus on primary goal
        path1 = {
            "sequence_id": 1,
            "rationale": f"Direct strategy: High progress ({d_t:.2f}), focus on completing primary goal efficiently via most direct path.",
            "action_sequence": self._generate_direct_actions(state, d_t, u_t),
            "confidence": 0.7 + 0.2 * d_t  # Higher confidence with higher progress
        }
        
        # Path 2: Proactive strategy - anticipate future needs
        path2 = {
            "sequence_id": 2,
            "rationale": f"Proactive strategy: Medium uncertainty ({u_t:.2f}), anticipate and address likely future needs to prevent dialogue breakdown.",
            "action_sequence": self._generate_proactive_actions(state, d_t, u_t, ρ_t),
            "confidence": 0.8 - 0.3 * u_t  # Lower confidence with higher uncertainty
        }
        
        # Path 3: Cautious strategy - verify information
        path3 = {
            "sequence_id": 3,
            "rationale": f"Cautious strategy: Slot dependency ({ρ_t:.2f}), verify existing information before requesting new slots to ensure consistency.",
            "action_sequence": self._generate_cautious_actions(state, d_t, ρ_t),
            "confidence": 0.6 + 0.3 * ρ_t  # Higher confidence with higher dependency
        }
        
        # Set immediate next actions
        for path in [path1, path2, path3]:
            if path["action_sequence"]:
                path["immediate_next_action"] = path["action_sequence"][0]
            else:
                path["immediate_next_action"] = ["general", "bye", ""]
        
        reasoning_paths = [path1, path2, path3]
        
        return reasoning_paths

    def _calculate_system1_confidence(self, response: str, cognitive_state: Tuple) -> float:
        """Calculate confidence score for System 1 prediction"""
        dialogue_progress, user_uncertainty, slot_dependency = cognitive_state
        
        # Base confidence from cognitive state
        base_confidence = 0.5 + 0.3 * dialogue_progress - 0.3 * user_uncertainty + 0.2 * slot_dependency
        base_confidence = max(0.1, min(0.95, base_confidence))
        
        # Adjust confidence based on response quality
        response_quality = 0.0
        
        # Check if response contains valid action format
        import re
        action_pattern = r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']\s*,\s*["\']([^"\']*)["\']\s*\]'
        
        if re.search(action_pattern, response):
            response_quality += 0.3  # Valid action format
        
        # Check if response contains action keywords
        action_keywords = ["request", "inform", "confirm", "book", "general"]
        for keyword in action_keywords:
            if keyword in response.lower():
                response_quality += 0.1
                break
        
        # Check response length and coherence
        if len(response.strip()) > 10 and len(response.strip()) < 200:
            response_quality += 0.2
        
        final_confidence = base_confidence + response_quality
        return max(0.1, min(0.95, final_confidence))

    def _system1_fallback_predict(self, state: Dict, cognitive_state: Tuple) -> Tuple[List, float]:
        """Fallback prediction method when System 1 model is not available"""
        dialogue_progress, user_uncertainty, slot_dependency = cognitive_state
        
        # Confidence based on cognitive state
        confidence = 0.5 + 0.3 * dialogue_progress - 0.3 * user_uncertainty + 0.2 * slot_dependency
        confidence = max(0.1, min(0.95, confidence))
        
        # Simple heuristic based on state for action selection
        action = []
        
        # Check if state has belief_state structure (MultiWOZ format)
        if 'belief_state' in state:
            belief_state = state['belief_state']
            
            # If we're in early stages and have high uncertainty, request more information
            if dialogue_progress < 0.3 and user_uncertainty > 0.5:
                # Find an unknown slot to request
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value == "?" or value is None or value == "" or value == "not mentioned":
                                action = [["request", domain, slot, ""]]
                                break
                        if action:
                            break
            
            # If we're in later stages and have low uncertainty, inform or book
            elif dialogue_progress > 0.7 and user_uncertainty < 0.3:
                # Find a filled slot to inform about
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value and value != "?" and value != "not mentioned":
                                action = [["inform", domain, slot, value]]
                                break
                        if action:
                            break
            
            # Default action if no specific condition is met
            if not action:
                # Find any slot to work with
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value and value != "?" and value != "not mentioned":
                                action = [["inform", domain, slot, value]]
                                break
                            elif value == "?" or value is None or value == "" or value == "not mentioned":
                                action = [["request", domain, slot, ""]]
                                break
                        if action:
                            break
        
        # Fallback for direct domain-slot structure or if no action was determined
        if not action:
            if 'belief_state' in state:
                # Try again with direct access
                for domain, slots in state['belief_state'].items():
                    if isinstance(slots, dict) and slots:
                        slot, value = next(iter(slots.items()))
                        if value and value != "?" and value != "not mentioned":
                            action = [["inform", domain, slot, value]]
                        else:
                            action = [["request", domain, slot, ""]]
                        break
            else:
                # Ultimate fallback
                action = [["general", "bye", ""]]
        
        return action, confidence

    def system2_reason(self, state: Dict, cognitive_state: Dict) -> List[Dict]:
        """
        System 2 reasoning (slow analytical response)
        Implements the reasoning process with prompts from the LaTeX paper
        
        Args:
            state: Current dialog state
            cognitive_state: Dict containing cognitive state components
            
        Returns:
            List of reasoning paths with actions and rationales
        """
        d_t = cognitive_state.get('dialogue_progress', 0.0)
        u_t = cognitive_state.get('user_uncertainty', 0.0)
        ρ_t = cognitive_state.get('slot_dependency', 0.0)
        
        # Construct the System 2 prompt using the helper method
        system2_prompt = self._construct_system2_prompt(state, cognitive_state)
        
        try:
            # Check if System 2 model is loaded
            if self.system2 is None:
                logging.warning("System 2 model not loaded, using fallback reasoning")
                return self._system2_fallback_reason(state, cognitive_state)
            
            # Tokenize and generate response
            tokenizer = self.system2['tokenizer']
            model = self.system2['model']
            
            inputs = tokenizer(system2_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the response to extract reasoning paths
            reasoning_paths = self._parse_system2_response(response)
            
            # If parsing fails, use fallback
            if not reasoning_paths:
                logging.warning("Failed to parse System 2 response, using fallback reasoning")
                return self._system2_fallback_reason(state, cognitive_state)
            
            logging.info(f"System 2 generated {len(reasoning_paths)} reasoning paths")
            
            # Add best decision to distillation buffer
            if hasattr(self, 'distillation_buffer') and self.is_train:
                best_action = None
                best_confidence = 0.0
                
                for path in reasoning_paths:
                    if path['confidence'] > best_confidence:
                        best_confidence = path['confidence']
                        best_action = path.get('immediate_next_action', [])
                
                if best_action and best_confidence > 0.9:  # Only add high-confidence decisions
                    # Add to distillation buffer
                    self.distillation_buffer.add(state, best_action, best_confidence)
                    logging.debug(f"Added System 2 decision to distillation buffer (confidence: {best_confidence:.3f})")
            
            return reasoning_paths
            
        except Exception as e:
            logging.error(f"Error in System 2 reasoning: {e}")
            # Fallback to heuristic reasoning
            return self._system2_fallback_reason(state, cognitive_state)
    
    def _generate_direct_actions(self, state: Dict, d_t: float, u_t: float) -> List:
        """Generate direct strategy actions focused on primary goal completion"""
        actions = []
        
        if 'belief_state' in state:
            belief_state = state['belief_state']
            
            # Direct strategy: focus on completing the most critical task
            if d_t < 0.5:  # Early stage: request missing information
                # Find the first unknown slot to request
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value in ["?", None, "", "not mentioned", "unknown"]:
                                actions.append(["request", domain, slot, ""])
                                break
                        if actions:
                            break
            else:  # Late stage: inform or book
                # Find a filled slot to inform about
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value and value not in ["?", "", "not mentioned", "unknown"]:
                                actions.append(["inform", domain, slot, value])
                                break
                        if actions:
                            break
        
        return actions if actions else [["general", "bye", ""]]
    
    def _generate_proactive_actions(self, state: Dict, d_t: float, u_t: float, ρ_t: float) -> List:
        """Generate proactive strategy actions anticipating future needs"""
        actions = []
        
        if 'belief_state' in state:
            belief_state = state['belief_state']
            
            # Proactive strategy: anticipate what might be needed next
            if u_t > 0.5:  # High uncertainty: provide reassurance
                actions.append(["general", "reassure", "I understand your needs, let me help you find the best option."])
                
                # Then request the most critical missing information
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value in ["?", None, "", "not mentioned", "unknown"]:
                                actions.append(["request", domain, slot, ""])
                                break
                        if len(actions) > 1:
                            break
            else:  # Low uncertainty: suggest next steps
                actions.append(["general", "suggest", "Based on what we have, I recommend the following:"])
                
                # Suggest completing the booking
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        actions.append(["book", domain, "", ""])
                        break
        
        return actions if actions else [["general", "bye", ""]]
    
    def _generate_cautious_actions(self, state: Dict, d_t: float, ρ_t: float) -> List:
        """Generate cautious strategy actions verifying information"""
        actions = []
        
        if 'belief_state' in state:
            belief_state = state['belief_state']
            
            # Cautious strategy: verify before proceeding
            if ρ_t > 0.5:  # High dependency: verify critical information
                # Find a filled slot to verify
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value and value not in ["?", "", "not mentioned", "unknown"]:
                                actions.append(["confirm", domain, slot, value])
                                break
                        if actions:
                            break
            else:  # Low dependency: proceed with standard verification
                actions.append(["general", "verify", "Let me confirm the information we have so far:"])
                
                # Then request missing information
                for domain, slots in belief_state.items():
                    if isinstance(slots, dict):
                        for slot, value in slots.items():
                            if value in ["?", None, "", "not mentioned", "unknown"]:
                                actions.append(["request", domain, slot, ""])
                                break
                        if len(actions) > 1:
                            break
        
        return actions if actions else [["general", "bye", ""]]

    def predict(self, state: Dict) -> List:
        """
        Main prediction function for DyBBT
        Implements the meta-controller mechanism to decide between System 1 and System 2
        
        Args:
            state: Current dialog state
            
        Returns:
            System action selected by the meta-controller
        """
        max_turns = self.config['dybbt']['max_turns']
        
        # Get current turn from session
        turn = 0
        if hasattr(self, 'sess') and self.sess and hasattr(self.sess, 'sys_agent'):
            turn = self.sess.sys_agent.turn
        elif hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'turn'):
            turn = self.agent.turn
        
        # Compute cognitive state c_t = [d_t, u_t, ρ_t]
        cognitive_state = self.compute_cognitive_state(state, turn, max_turns)
        
        # Get System 1 prediction (fast intuitive response)
        system1_action, system1_confidence = self.system1_predict(state, cognitive_state)
        
        # Meta-controller decision based on LaTeX paper algorithm
        should_activate_system2 = self.should_activate_system2(cognitive_state, system1_confidence)
        
        if should_activate_system2:
            logging.info(f"Meta-controller: Activating System 2. "
                        f"Cognitive state: {cognitive_state}, "
                        f"S1 confidence: {system1_confidence:.3f}")
            
            # Get System 2 reasoning (slow analytical response)
            reasoning_paths = self.system2_reason(state, cognitive_state)
            
            # Select best immediate action from reasoning paths
            best_action = None
            best_confidence = 0.0
            best_path_id = 0
            
            for path in reasoning_paths:
                # Use path confidence as the selection criterion
                if path['confidence'] > best_confidence:
                    best_confidence = path['confidence']
                    best_action = path['immediate_next_action']
                    best_path_id = path['sequence_id']
            
            # If no best action found, fall back to System 1
            if best_action is None:
                logging.info("Meta-controller: No best action from System 2, falling back to System 1")
                return system1_action
            
            logging.info(f"Meta-controller: Selected System 2 Path {best_path_id}. "
                        f"Action: {best_action}, Confidence: {best_confidence:.3f}")
            return best_action
        else:
            logging.info(f"Meta-controller: Using System 1 for fast response. "
                        f"Action: {system1_action}, Confidence: {system1_confidence:.3f}")
            return system1_action

    def init_session(self, agent=None, **kwargs):
        """Reset session-specific state"""
        self.cognitive_state_counts = {}
        self.current_cognitive_state = None
        # Store agent reference for turn access
        self.agent = agent
        # Also accept agent from kwargs for compatibility
        if agent is None and 'agent' in kwargs:
            self.agent = kwargs['agent']

    def update(self, *args, **kwargs):
        """Placeholder for training update - would be implemented for PPO"""
        pass

    def save(self, directory: str, addition: str = ""):
        """Save model weights"""
        # Placeholder for model saving
        pass

    def load(self, filename: str):
        """Load model weights"""
        # Placeholder for model loading
        pass

    @classmethod
    def from_pretrained(cls, config: Dict, model_file: str = "", is_train: bool = False):
        """Create DyBBT instance from pretrained weights"""
        instance = cls(config, is_train)
        if model_file:
            instance.load(model_file)
        return instance
    
    def distill_system2_knowledge(self) -> None:
        """
        Distill high-quality System 2 decisions to improve System 1
        This method should be called periodically during training
        """
        if not self.is_train or not hasattr(self, 'distillation_buffer'):
            return
            
        buffer_size = self.distillation_buffer.size()
        if buffer_size == 0:
            logging.info("Distillation buffer is empty, skipping distillation")
            return
        
        logging.info(f"Starting knowledge distillation from {buffer_size} System 2 decisions")
        
        try:
            # Sample batch from distillation buffer
            batch = self.distillation_buffer.sample_batch(batch_size=4)
            if not batch:
                logging.info("No samples available for distillation")
                return
            
            # Prepare for LoRA fine-tuning
            if self.system1 is None or self.system1['model'] is None:
                logging.warning("System 1 model not available for distillation")
                return
                
            model = self.system1['model']
            tokenizer = self.system1['tokenizer']
            
            # Check if model is a PeftModel (has LoRA adapters)
            if not isinstance(model, PeftModel):
                logging.warning("System 1 model is not a PeftModel, cannot perform LoRA fine-tuning")
                return
            
            # Set model to training mode
            model.train()
            
            # Perform LoRA fine-tuning on sampled batch
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            total_loss = 0.0
            
            for state, target_action, confidence in batch:
                # Construct prompt for System 1
                belief_state_str = json.dumps(state.get('belief_state', {}), ensure_ascii=False, indent=2)
                
                # Compute cognitive state for this sample
                turn = 0  # Placeholder, would need to get actual turn from state
                max_turns = self.config['dybbt']['max_turns']
                cognitive_state = self.compute_cognitive_state(state, turn, max_turns)
                d_t, u_t, ρ_t = cognitive_state
                
                system1_prompt = f"""
You are the fast intuitive component (System 1) of a dialogue system. Your task is to generate the most appropriate next action based on the current cognitive state.

**Current Cognitive State:**
- Dialogue Progress (d_t): {d_t:.3f}
- User Uncertainty (u_t): {u_t:.3f}  
- Slot Dependency (ρ_t): {ρ_t:.3f}

**Current Belief State:**
{belief_state_str}

**Instructions:**
Generate the most appropriate next system action in the format: [action_type, domain, slot, value]

Available action types: request, inform, confirm, book, general

Output ONLY the action in the specified format, nothing else.
"""
                
                # Tokenize inputs and targets
                inputs = tokenizer(system1_prompt, return_tensors="pt", truncation=True, max_length=1024)
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                
                # Format target action for training
                target_text = json.dumps({
                    "action": target_action,
                    "confidence": confidence
                }, ensure_ascii=False)
                
                target_ids = tokenizer.encode(target_text, return_tensors="pt").to(DEVICE)
                
                # Forward pass
                outputs = model(**inputs, labels=target_ids)
                loss = outputs.loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # Set model back to evaluation mode
            model.eval()
            
            avg_loss = total_loss / len(batch)
            logging.info(f"Knowledge distillation completed using {len(batch)} samples. Average loss: {avg_loss:.4f}")
            
        except Exception as e:
            logging.error(f"Error during knowledge distillation: {e}")
            # Make sure to set model back to eval mode even if error occurs
            if self.system1 and self.system1['model']:
                self.system1['model'].eval()
    
    def get_distillation_buffer_size(self) -> int:
        """
        Get current size of distillation buffer
        
        Returns:
            Number of entries in distillation buffer
        """
        if hasattr(self, 'distillation_buffer'):
            return self.distillation_buffer.size()
        return 0
    
    def clear_distillation_buffer(self) -> None:
        """Clear all entries from distillation buffer"""
        if hasattr(self, 'distillation_buffer'):
            self.distillation_buffer.clear()
