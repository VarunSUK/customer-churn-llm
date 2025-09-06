"""
LLM client for OpenAI/LiteLLM integration.
Handles prompt execution and response parsing.
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
try:
    import openai
except ImportError:
    openai = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from models import ActionPlan, MessageDraftResponse


class LLMClient:
    """Client for LLM interactions."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-1.5-flash", provider: str = "gemini"):
        """
        Initialize the LLM client.
        
        Args:
            api_key: API key (if None, will try to get from environment)
            model: Model to use for LLM calls
            provider: LLM provider ("gemini", "deepseek", or "openai")
        """
        self.provider = provider
        self.model = model
        
        if provider == "gemini":
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                print("Warning: No Gemini API key found. LLM features will not work.")
                print("Set GEMINI_API_KEY environment variable or pass api_key parameter.")
                self.client = None
            else:
                if genai is None:
                    print("Warning: Google Generative AI library not installed. Please install with: pip install google-generativeai")
                    self.client = None
                else:
                    # Configure Gemini client
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(model)
        elif provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                print("Warning: No DeepSeek API key found. LLM features will not work.")
                print("Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")
                self.client = None
            else:
                if openai is None:
                    print("Warning: OpenAI library not installed. Please install with: pip install openai")
                    self.client = None
                else:
                    # Configure OpenAI client for DeepSeek
                    self.client = openai.OpenAI(
                        api_key=self.api_key,
                        base_url="https://api.deepseek.com"
                    )
        else:  # OpenAI
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key or openai is None:
                print("Warning: No OpenAI API key found or OpenAI not installed. LLM features will not work.")
                print("Set OPENAI_API_KEY environment variable or pass api_key parameter.")
                self.client = None
            else:
                self.client = openai.OpenAI(api_key=self.api_key)
    
    def _load_prompt_template(self, prompt_name: str) -> str:
        """Load a prompt template from the prompts directory."""
        prompt_path = Path(__file__).parent.parent / "prompts" / f"{prompt_name}.md"
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _format_prompt(self, template: str, **kwargs) -> str:
        """Format a prompt template with provided variables."""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required variable in prompt: {e}")
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling various formats."""
        # Try to find JSON in the response
        json_patterns = [
            r'```json\s*(\{.*?\})\s*```',  # JSON in code block
            r'```\s*(\{.*?\})\s*```',      # JSON in generic code block
            r'(\{.*?\})',                  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue
        
        # If no JSON found, try to parse the entire response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Could not extract valid JSON from response: {response[:200]}...")
    
    def plan_action(self, 
                   churn_probability: float,
                   top_reasons: List[Dict[str, Any]],
                   customer_segment: Optional[str] = None,
                   uplift_by_action: Optional[Dict[str, float]] = None,
                   rag_summary: Optional[str] = None) -> ActionPlan:
        """
        Generate an action plan using the LLM.
        
        Args:
            churn_probability: Customer churn probability
            top_reasons: List of top contributing features
            customer_segment: Customer segment (optional)
            uplift_by_action: Historical uplift data (optional)
            rag_summary: Recent issues summary (optional)
            
        Returns:
            ActionPlan object
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Check API key.")
        
        # Load and format the prompt
        template = self._load_prompt_template("action_planner")
        
        # Format the top reasons for the prompt
        reasons_text = "\n".join([f"- {r['feature']}: {r['impact']:.3f}" for r in top_reasons])
        
        # Format uplift data if provided
        uplift_text = ""
        if uplift_by_action:
            uplift_text = "\n".join([f"- {action}: {uplift:.3f}" for action, uplift in uplift_by_action.items()])
        
        prompt = self._format_prompt(
            template,
            churn_probability=churn_probability,
            top_reasons=reasons_text,
            customer_segment=customer_segment or "unknown",
            uplift_by_action=uplift_text or "No historical data available",
            rag_summary=rag_summary or "No recent issues data available"
        )
        
        try:
            if self.provider == "gemini":
                # Gemini API call
                response = self.client.generate_content(
                    f"You are a customer retention specialist AI. Always respond with valid JSON only.\n\n{prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=500
                    )
                )
                response_text = response.text
            else:
                # OpenAI/DeepSeek API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a customer retention specialist AI. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            action_data = self._extract_json_from_response(response_text)
            
            # Validate and create ActionPlan
            return ActionPlan(**action_data)
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate action plan: {e}")
    
    def draft_message(self,
                     customer_name: str,
                     action_type: str,
                     message_outline: List[str],
                     churn_probability: float,
                     top_reasons: List[Dict[str, Any]]) -> MessageDraftResponse:
        """
        Draft a customer message using the LLM.
        
        Args:
            customer_name: Customer name
            action_type: Action type
            message_outline: Key points to include
            churn_probability: Customer churn probability
            top_reasons: Top contributing features
            
        Returns:
            MessageDraftResponse object
        """
        if not self.client:
            raise RuntimeError("LLM client not initialized. Check API key.")
        
        # Load and format the prompt
        template = self._load_prompt_template("message_generator")
        
        # Format the top reasons for the prompt
        reasons_text = "\n".join([f"- {r['feature']}: {r['impact']:.3f}" for r in top_reasons])
        
        # Format message outline
        outline_text = "\n".join([f'"{point}"' for point in message_outline])
        
        prompt = self._format_prompt(
            template,
            customer_name=customer_name,
            action_type=action_type,
            message_outline=outline_text,
            churn_probability=churn_probability,
            top_reasons=reasons_text
        )
        
        try:
            if self.provider == "gemini":
                # Gemini API call
                response = self.client.generate_content(
                    f"You are a customer communication specialist AI. Draft empathetic, professional messages.\n\n{prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=300
                    )
                )
                message_text = response.text.strip()
            else:
                # OpenAI/DeepSeek API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a customer communication specialist AI. Draft empathetic, professional messages."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                message_text = response.choices[0].message.content.strip()
            
            # Count words
            word_count = len(message_text.split())
            
            # Determine tone (simple heuristic)
            tone = "empathetic" if any(word in message_text.lower() for word in ["understand", "apologize", "sorry", "concern"]) else "professional"
            
            # Extract call to action (simple heuristic)
            cta = "contact us" if "contact" in message_text.lower() else "respond" if "reply" in message_text.lower() else "take action"
            
            return MessageDraftResponse(
                message=message_text,
                word_count=word_count,
                tone=tone,
                call_to_action=cta
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to draft message: {e}")
    
    def is_available(self) -> bool:
        """Check if the LLM client is available and configured."""
        return self.client is not None and self.api_key is not None


# Global LLM client instance with Gemini
llm_client = LLMClient(provider="gemini", model="gemini-1.5-flash")
