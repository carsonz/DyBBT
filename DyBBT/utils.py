#!/usr/bin/env python3
"""
Utility functions and utility classes
"""

from typing import Dict, List
import random
import numpy as np
import json
from convlab.policy.vector.vector_base import VectorBase
from convlab.util.multiwoz.lexicalize import deflat_da, lexicalize_da

def extract_domains_from_action(dialogue_acts: Dict) -> List[str]:
    """Extract domains from user actions"""
    domains = set()
    # Traverse all action types (categorical, non-categorical, binary)
    for act_type in ['categorical', 'non-categorical', 'binary']:
        if act_type in dialogue_acts:
            # Traverse all actions under this type
            for act in dialogue_acts[act_type]:
                # Check if act is a dictionary and contains domain key
                if isinstance(act, dict) and 'domain' in act:
                    domains.add(act['domain'])
    return list(domains)

def get_domains_from_action(action: List) -> List[str]:
    """Extract domains from actions"""
    domains = set()
    for act in action:
        if len(act) > 1:  # Ensure act has at least 2 elements
            domains.add(act[1])  # Domain information is at position act[1]
    return list(domains)

def get_domains_belief_state(domains: List[str], belief_state: Dict) -> Dict:
    """Filter belief state by domains"""
    filtered_belief_state = {}
    for domain in domains:
        if domain in belief_state:
            filtered_belief_state[domain] = belief_state[domain]
    return filtered_belief_state

def get_available_actions(vector_base: VectorBase, domains: List[str], belief_state: Dict) -> List[str]:
    """Get available actions based on domains and belief state"""
    # Initialize domain activation dictionary
    domain_active_dict = vector_base.init_domain_active_dict()
    
    # Activate mentioned domains
    for domain in domains:
        if domain in domain_active_dict:
            domain_active_dict[domain] = True
    
    # Activate domains with values
    for domain in belief_state:
        if domain in domain_active_dict and belief_state[domain]:
            domain_active_dict[domain] = True
    
    # Compute domain mask
    domain_mask = vector_base.compute_domain_mask(domain_active_dict)
    
    # Get available actions
    available_actions = []
    for i in range(vector_base.da_dim):
        if domain_mask[i] == 0:  # Actions not masked are available
            action = vector_base.vec2act[i]
            # Check if action's domain is in current domain list
            if action[0] in domains:
                # Keep only first three elements of action
                modified_action = list(action)[:3]
                available_actions.append(tuple(modified_action))
    
    # Remove duplicate actions
    available_actions = list(set(available_actions))
    
    # Filter available_actions based on slots with values in belief_state
    filtered_actions = []
    for action in available_actions:
        domain, intent, slot = action
        # Check if current action's slot has value in belief_state
        if domain in belief_state and slot in belief_state[domain] and belief_state[domain][slot]:
            # If has value, skip this action
            continue
        else:
            # If no value, keep this action
            filtered_actions.append(action)
    
    return filtered_actions

def parse_model_output(output_text):
    """Parse model output"""
    try:
        # Extract JSON part
        end = output_text.rfind('}') + 1
        start = output_text.rfind('{', 0, end)  # Search for { from end position forward
        
        if start != -1 and end > start:
            json_str = output_text[start:end]
            result = json.loads(json_str)
            if 'action' in result:
                # Remove leading and trailing spaces from all fields in action
                action = result['action']
                if isinstance(action, list):
                    # Traverse double array, remove leading and trailing spaces from each string
                    cleaned_action = []
                    for sublist in action:
                        if isinstance(sublist, list):
                            cleaned_sublist = []
                            for item in sublist:
                                if isinstance(item, str):
                                    cleaned_sublist.append(item.strip())
                                else:
                                    cleaned_sublist.append(item)
                            cleaned_action.append(cleaned_sublist)
                        else:
                            cleaned_action.append(sublist)
                    return cleaned_action
                return action
            else:
                return None
    except Exception as e:
        #print(output_text)
        return None
    return None
    
class CustomVectorBase(VectorBase):
    """Custom VectorBase class, implements get_state_dim method"""
    
    def get_state_dim(self):
        '''
        Compute the state dimension for the policy input
        '''
        self.state_dim = 0
        # A default state_dim value can be set here
        # Or calculate state_dim based on actual requirements
        # Since we only care about actions, set it to 0 here
        return self.state_dim
    
    def add_values_to_act(self, domain, intent, slot):
        """
        The ontology does not contain information about the value of an act. This method will add the value and
        is based on how it is created in MultiWOZ. This might need to be changed for other datasets such as SGD.
        """

        if intent == 'request':
            return (domain, intent, slot, "?")

        if slot == '':
            return (domain, intent, "none", "none")

        if intent in ['recommend', 'select', 'inform']:
            return (domain, intent, slot, str(random.randint(1, 4)))
        else:
            return (domain, intent, slot, "1")

    def form_actions(self, action_array):
        try:
            act_array = [self.add_values_to_act(action_array[i][0], action_array[i][1], action_array[i][2]) for i in range(len(action_array))]
        except:
            act_array = []

        if len(act_array) == 0:
            if self.reqinfo_filler_action:
                act_array.append(("general", "reqinfo", "none", "none"))
            else:
                act_array.append(("general", "reqmore", "none", "none"))
        
        action = deflat_da(act_array)

        entities = {}
        for domint in action:
            domain, intent = domint
            if domain not in entities and domain not in ['general']:
                entities[domain] = self.dbquery_domain(domain)

        # From db query find which slot causes no_offer
        nooffer = [domint for domint in action if 'nooffer' in domint[1]]
        for domint in nooffer:
            domain, intent = domint
            slot = self.find_nooffer_slot(domain)
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        # Randomly select booking constraint "causing" no_book
        nobook = [domint for domint in action if 'nobook' in domint[1]]
        for domint in nobook:
            domain, intent = domint
            if domain in self.state:
                slots = self.state[domain]
                slots = [slot for slot, i in slots.items()
                         if i and 'book' in slot]
                slots.append('none')
                slot = np.random.choice(slots)
            else:
                slot = 'none'
            action[domint] = [[slot, '1']
                              ] if slot != 'none' else [[slot, 'none']]

        if self.always_inform_booking_reference:
            action = self.add_booking_reference(action)

        # When there is a INFORM(1 name) or OFFER(multiple) action then inform the name
        if self.use_add_name:
            action = self.add_name(action)

        for key in action.keys():
            index = -1
            for [item, idx] in action[key]:
                if index != -1 and index != idx and idx != '?':
                    pass
                    # logging.debug(
                    #    "System is likely refering multiple entities within this turn")
                    # logging.debug(action[key])
                index = idx
        # Handle case where domain might not exist in state
        try:
            action = lexicalize_da(action, entities, self.state, self.requestable)
        except KeyError as e:
            print(f"Warning: KeyError in lexicalize_da for domain {e}. Using empty state for this domain.")
            # Create a copy of state and add missing domain
            safe_state = self.state.copy()
            missing_domain = str(e).strip("'")
            safe_state[missing_domain] = {}
            action = lexicalize_da(action, entities, safe_state, self.requestable)

        if not self.use_none:
            # replace all occurences of "none" with an empty string ""
            f = lambda x: x if x != "none" else ""
            action = [[f(x) for x in a_list] for a_list in action]
            #action = [[ for a_tuple in a_list] for a_list in action]

        return action
    