#!/usr/bin/env python3
"""
REACT-Style Reverse Image Search Validator
Enhanced reverse search with think → act process and clue validation
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import openai
from dataclasses import dataclass, asdict

# Setup logging
logger = logging.getLogger('REACTSearchValidator')

@dataclass
class ClueValidationStep:
    """Represents a single validation step in the REACT process"""
    step_type: str  # "think", "act", "check", "validate"
    timestamp: str
    content: str
    confidence: float
    reasoning: str
    action_taken: Optional[str] = None
    validation_result: Optional[bool] = None

@dataclass 
class ClueValidationResult:
    """Result of clue validation with REACT steps"""
    clue: str
    original_confidence: float
    validated_confidence: float
    is_reliable: bool
    validation_steps: List[ClueValidationStep]
    cross_validation_score: float
    potential_misleading_factors: List[str]
    recommendation: str

class REACTSearchValidator:
    """REACT-style validator for reverse image search clues"""
    
    def __init__(self, openai_client=None, display_process: bool = True):
        self.openai_client = openai_client
        self.display_process = display_process
        self.validation_history = []
        
    def display_step(self, step: str, content: str, confidence: Optional[float] = None):
        """Display the think → act process to user"""
        if not self.display_process:
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        confidence_str = f" (confidence: {confidence:.2f})" if confidence else ""
        
        icons = {
            "think": "[THINK]",
            "act": "[ACT]",
            "check": "[CHECK]", 
            "validate": "[VALIDATE]",
            "warn": "[WARN]",
            "error": "[ERROR]"
        }
        
        icon = icons.get(step, "💭")
        print(f"{icon} [{timestamp}] {step.upper()}: {content}{confidence_str}")
        
    async def validate_clues_with_react(
        self, 
        clues: List[Dict[str, Any]], 
        original_image_context: str,
        source_urls: List[str]
    ) -> List[ClueValidationResult]:
        """
        Validate extracted clues using REACT methodology
        
        Args:
            clues: List of extracted geographic clues with metadata
            original_image_context: Context about the original image
            source_urls: URLs where clues were found
            
        Returns:
            List of validation results with REACT steps
        """
        
        self.display_step("think", f"Starting REACT validation for {len(clues)} clues")
        
        validation_results = []
        
        for i, clue_data in enumerate(clues):
            self.display_step("act", f"Processing clue {i+1}: {clue_data.get('clue', '')[:50]}...")
            
            # Step 1: REASONING - Analyze the clue
            reasoning_result = await self._reasoning_step(clue_data, original_image_context)
            
            # Step 2: ACTING - Cross-validate with other sources  
            action_result = await self._action_step(clue_data, clues, source_urls)
            
            # Step 3: CHECKING - Evaluate consistency and reliability
            checking_result = await self._checking_step(clue_data, reasoning_result, action_result)
            
            # Step 4: THINKING AGAIN - Final validation decision
            final_result = await self._thinking_step(clue_data, reasoning_result, action_result, checking_result)
            
            validation_results.append(final_result)
            
            # Brief pause between clues
            await asyncio.sleep(0.5)
        
        # Final summary
        reliable_clues = len([r for r in validation_results if r.is_reliable])
        self.display_step("validate", f"Validation complete: {reliable_clues}/{len(clues)} clues deemed reliable")
        
        return validation_results
    
    async def _reasoning_step(self, clue_data: Dict, image_context: str) -> ClueValidationStep:
        """Step 1: Reasoning - Analyze the clue's plausibility"""
        
        clue = clue_data.get('clue', '')
        source_url = clue_data.get('source_url', '')
        
        self.display_step("think", f"Analyzing plausibility of: '{clue}'")
        
        if not self.openai_client:
            # Fallback reasoning without GPT
            reasoning = "Basic heuristic analysis applied"
            confidence = 0.5  # Medium confidence
            step_content = f"Applied basic validation to clue: {clue}"
        else:
            # GPT-powered reasoning
            reasoning_prompt = f"""
            Analyze this geographic clue for plausibility and potential issues:
            
            Clue: "{clue}"
            Source URL: {source_url}
            Image Context: {image_context}
            
            Consider:
            1. Is this clue specific enough to be useful?
            2. Could this clue be misleading (e.g., from unrelated content)?
            3. Does it match the image context?
            4. What are potential reliability issues?
            
            Respond in JSON format:
            {{
                "plausibility_score": 0.0-1.0,
                "specificity": "high/medium/low",
                "potential_issues": ["list", "of", "concerns"],
                "reasoning": "detailed analysis",
                "reliability_factors": ["positive", "factors"],
                "warning_signs": ["negative", "factors"]
            }}
            """
            
            try:
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": reasoning_prompt}],
                    max_tokens=500,
                    temperature=0.1
                )
                
                result = json.loads(response.choices[0].message.content.strip())
                confidence = result.get('plausibility_score', 0.5)
                reasoning = result.get('reasoning', 'GPT analysis completed')
                step_content = f"Plausibility: {confidence:.2f}, Issues: {len(result.get('potential_issues', []))}"
                
                if result.get('warning_signs'):
                    self.display_step("warn", f"Warning signs detected: {', '.join(result['warning_signs'][:2])}")
                    
            except Exception as e:
                logger.warning(f"GPT reasoning failed: {e}")
                reasoning = f"GPT analysis failed: {str(e)}"
                confidence = 0.3  # Lower confidence due to failure
                step_content = f"Fallback analysis for: {clue}"
        
        self.display_step("think", step_content, confidence)
        
        return ClueValidationStep(
            step_type="think",
            timestamp=datetime.now().isoformat(),
            content=step_content,
            confidence=confidence,
            reasoning=reasoning
        )
    
    async def _action_step(self, clue_data: Dict, all_clues: List[Dict], source_urls: List[str]) -> ClueValidationStep:
        """Step 2: Acting - Cross-validate with other clues and sources"""
        
        clue = clue_data.get('clue', '')
        self.display_step("act", f"Cross-validating: '{clue[:30]}...'")
        
        # Count supporting clues
        supporting_clues = 0
        contradicting_clues = 0
        
        for other_clue_data in all_clues:
            if other_clue_data == clue_data:
                continue
                
            other_clue = other_clue_data.get('clue', '')
            
            # Simple similarity check (can be enhanced with NLP)
            if self._clues_are_related(clue, other_clue):
                if self._clues_are_contradictory(clue, other_clue):
                    contradicting_clues += 1
                else:
                    supporting_clues += 1
        
        # Calculate cross-validation score
        total_related = supporting_clues + contradicting_clues
        if total_related == 0:
            cross_val_score = 0.5  # Neutral if no related clues
        else:
            cross_val_score = supporting_clues / total_related
        
        step_content = f"Cross-validation: {supporting_clues} supporting, {contradicting_clues} contradicting"
        self.display_step("act", step_content, cross_val_score)
        
        return ClueValidationStep(
            step_type="act", 
            timestamp=datetime.now().isoformat(),
            content=step_content,
            confidence=cross_val_score,
            reasoning=f"Found {total_related} related clues for comparison",
            action_taken="cross_validation"
        )
    
    async def _checking_step(self, clue_data: Dict, reasoning: ClueValidationStep, action: ClueValidationStep) -> ClueValidationStep:
        """Step 3: Checking - Evaluate overall consistency"""
        
        clue = clue_data.get('clue', '')
        self.display_step("check", f"Evaluating consistency for: '{clue[:30]}...'")
        
        # Combine reasoning and cross-validation scores
        combined_confidence = (reasoning.confidence * 0.6) + (action.confidence * 0.4)
        
        # Identify potential issues
        potential_issues = []
        
        if reasoning.confidence < 0.3:
            potential_issues.append("Low plausibility score")
        
        if action.confidence < 0.3:
            potential_issues.append("Poor cross-validation")
            
        if combined_confidence < 0.4:
            potential_issues.append("Overall low confidence")
            
        # Check for specificity
        if len(clue.split()) < 2:
            potential_issues.append("Overly generic clue")
            
        step_content = f"Consistency check: {combined_confidence:.2f}, Issues: {len(potential_issues)}"
        
        if potential_issues:
            self.display_step("warn", f"Issues found: {', '.join(potential_issues[:2])}")
            
        self.display_step("check", step_content, combined_confidence)
        
        return ClueValidationStep(
            step_type="check",
            timestamp=datetime.now().isoformat(),
            content=step_content,
            confidence=combined_confidence,
            reasoning=f"Combined analysis of reasoning and cross-validation. Issues: {potential_issues}",
            validation_result=combined_confidence >= 0.5
        )
    
    async def _thinking_step(self, clue_data: Dict, reasoning: ClueValidationStep, 
                           action: ClueValidationStep, checking: ClueValidationStep) -> ClueValidationResult:
        """Step 4: Final thinking - Make validation decision"""
        
        clue = clue_data.get('clue', '')
        original_confidence = clue_data.get('confidence', 0.5)
        
        self.display_step("think", f"Final evaluation for: '{clue[:30]}...'")
        
        # Ensure original_confidence is a float
        if isinstance(original_confidence, str):
            # Try to parse confidence levels like "high", "medium", "low"
            confidence_mapping = {"high": 0.8, "medium": 0.6, "low": 0.4, "unknown": 0.5}
            original_confidence = confidence_mapping.get(original_confidence.lower(), 0.5)
        
        # Determine if clue is reliable
        is_reliable = checking.validation_result and checking.confidence >= 0.5
        
        # Calculate final validated confidence
        validated_confidence = min(float(original_confidence), checking.confidence)
        
        # Generate recommendation
        if is_reliable:
            if validated_confidence >= 0.7:
                recommendation = "High confidence - use as primary clue"
                self.display_step("validate", f"RELIABLE: {clue[:40]}...")
            else:
                recommendation = "Medium confidence - use as supporting evidence" 
                self.display_step("validate", f"RELIABLE (moderate): {clue[:40]}...")
        else:
            recommendation = "Low confidence - consider excluding or verify independently"
            self.display_step("warn", f"UNRELIABLE: {clue[:40]}...")
        
        # Identify potential misleading factors
        misleading_factors = []
        if reasoning.confidence < 0.4:
            misleading_factors.append("Implausible geographic information")
        if action.confidence < 0.4:
            misleading_factors.append("Contradicted by other sources")
        if validated_confidence < original_confidence * 0.7:
            misleading_factors.append("Significant confidence reduction after validation")
            
        return ClueValidationResult(
            clue=clue,
            original_confidence=original_confidence,
            validated_confidence=validated_confidence,
            is_reliable=is_reliable,
            validation_steps=[reasoning, action, checking],
            cross_validation_score=action.confidence,
            potential_misleading_factors=misleading_factors,
            recommendation=recommendation
        )
    
    def _clues_are_related(self, clue1: str, clue2: str) -> bool:
        """Check if two clues are geographically related"""
        # Simple keyword overlap check - can be enhanced
        words1 = set(clue1.lower().split())
        words2 = set(clue2.lower().split())
        
        # Geographic terms that suggest relatedness
        geo_terms = {'country', 'city', 'state', 'region', 'location', 'coordinates', 'latitude', 'longitude'}
        
        # Check for shared geographic terms or place names
        common_words = words1.intersection(words2)
        geo_overlap = common_words.intersection(geo_terms)
        
        # Related if they share geographic terms or have significant word overlap
        return len(geo_overlap) > 0 or len(common_words) >= 2
    
    def _clues_are_contradictory(self, clue1: str, clue2: str) -> bool:
        """Check if two clues contradict each other"""
        # Simple contradiction detection - can be enhanced
        clue1_lower = clue1.lower()
        clue2_lower = clue2.lower()
        
        # Look for contradictory country/region mentions
        countries1 = self._extract_countries(clue1_lower)
        countries2 = self._extract_countries(clue2_lower)
        
        if countries1 and countries2:
            return not bool(countries1.intersection(countries2))
            
        return False
    
    def _extract_countries(self, text: str) -> set:
        """Extract country names from text - simplified version"""
        common_countries = {
            'usa', 'united states', 'america', 'canada', 'mexico', 'brazil', 'argentina',
            'uk', 'united kingdom', 'england', 'scotland', 'ireland', 'france', 'germany', 'spain', 'italy',
            'russia', 'china', 'japan', 'korea', 'india', 'thailand', 'vietnam', 'australia', 'new zealand'
        }
        
        found_countries = set()
        for country in common_countries:
            if country in text:
                found_countries.add(country)
                
        return found_countries
    
    def get_validation_summary(self, validation_results: List[ClueValidationResult]) -> Dict[str, Any]:
        """Generate a summary of the validation process"""
        
        total_clues = len(validation_results)
        reliable_clues = len([r for r in validation_results if r.is_reliable])
        avg_confidence_before = sum(r.original_confidence for r in validation_results) / total_clues if total_clues > 0 else 0
        avg_confidence_after = sum(r.validated_confidence for r in validation_results) / total_clues if total_clues > 0 else 0
        
        misleading_factors = []
        for result in validation_results:
            misleading_factors.extend(result.potential_misleading_factors)
        
        # Count most common misleading factors
        factor_counts = {}
        for factor in misleading_factors:
            factor_counts[factor] = factor_counts.get(factor, 0) + 1
            
        return {
            "total_clues_validated": total_clues,
            "reliable_clues": reliable_clues,
            "reliability_rate": reliable_clues / total_clues if total_clues > 0 else 0,
            "avg_confidence_before_validation": avg_confidence_before,
            "avg_confidence_after_validation": avg_confidence_after,
            "confidence_improvement": avg_confidence_after - avg_confidence_before,
            "top_misleading_factors": sorted(factor_counts.items(), key=lambda x: x[1], reverse=True)[:3],
            "validation_timestamp": datetime.now().isoformat(),
            "total_validation_steps": sum(len(r.validation_steps) for r in validation_results)
        }

# Helper function to integrate with existing reverse search
async def enhance_reverse_search_with_react(
    search_results: Dict[str, Any], 
    original_image_path: str,
    openai_client=None,
    display_process: bool = True
) -> Dict[str, Any]:
    """
    Enhance existing reverse search results with REACT validation
    
    Args:
        search_results: Results from existing reverse image search
        original_image_path: Path to the original image
        openai_client: OpenAI client for GPT analysis
        display_process: Whether to display the think → act process
        
    Returns:
        Enhanced results with REACT validation
    """
    
    validator = REACTSearchValidator(openai_client, display_process)
    
    # Extract clues from existing results
    clues = []
    source_urls = []
    
    if "results" in search_results:
        for result in search_results["results"]:
            if result.get("success") and result.get("geographic_clues"):
                source_url = result.get("source_url", "")
                source_urls.append(source_url)
                
                for clue in result["geographic_clues"]:
                    clues.append({
                        "clue": clue,
                        "source_url": source_url,
                        "confidence": result.get("confidence", 0.5),
                        "original_source": result.get("domain", "unknown")
                    })
    
    if not clues:
        return {
            **search_results,
            "react_validation": {
                "status": "no_clues_to_validate",
                "message": "No geographic clues found in search results"
            }
        }
    
    # Perform REACT validation
    image_context = f"Original image: {original_image_path}"
    validation_results = await validator.validate_clues_with_react(clues, image_context, source_urls)
    
    # Generate summary
    summary = validator.get_validation_summary(validation_results)
    
    # Update search results with validation
    enhanced_results = {
        **search_results,
        "react_validation": {
            "status": "completed",
            "summary": summary,
            "validation_results": [asdict(result) for result in validation_results],
            "reliable_clues": [result.clue for result in validation_results if result.is_reliable],
            "unreliable_clues": [result.clue for result in validation_results if not result.is_reliable]
        }
    }
    
    return enhanced_results

if __name__ == "__main__":
    # Example usage
    print("🧠 REACT-Style Reverse Image Search Validator")
    print("=" * 50)
    print("This module enhances reverse image search with:")
    print("• Reasoning - Analyze clue plausibility")  
    print("• Acting - Cross-validate with other sources")
    print("• Checking - Evaluate consistency") 
    print("• Validating - Make final reliability decision")
    print("\nIntegrate with existing search using enhance_reverse_search_with_react()")
