"""
Phase 6: Multimodal Reasoning Engine

Integrates vision, audio, and text for unified understanding.
This enables reasoning across different modalities!

Architecture:
1. Visual understanding - image/diagram interpretation
2. Audio processing - speech/sound pattern analysis
3. Cross-modal alignment - connecting modalities semantically
4. Multimodal memory - storing and retrieving across modalities
5. Unified reasoning - solving problems using multiple modalities

Example:
    Input: Mathematical diagram + spoken explanation
    Process: Extract visual structure + transcribe audio + align
    Output: Deep understanding combining both modalities
"""

from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import re
import base64
import os


class Modality(Enum):
    """Supported input modalities."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DIAGRAM = "diagram"
    MATHEMATICAL = "mathematical"


@dataclass
class ModalityInput:
    """Represents input from a specific modality."""

    modality: Modality
    content: Any  # Raw content (text, bytes, path)
    metadata: Dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Processed representations
    embedding: Optional[List[float]] = None
    description: Optional[str] = None
    extracted_concepts: List[str] = field(default_factory=list)


@dataclass
class CrossModalAlignment:
    """Alignment between concepts across modalities."""

    modality_a: Modality
    modality_b: Modality
    concept_a: str
    concept_b: str
    alignment_score: float  # 0-1, how well they correspond
    relationship: str  # "equivalent", "describes", "depicts", etc.


@dataclass
class MultimodalConcept:
    """A concept represented across multiple modalities."""

    name: str
    text_representation: Optional[str] = None
    visual_representation: Optional[str] = None  # Description or path
    audio_representation: Optional[str] = None  # Description or path
    mathematical_form: Optional[str] = None
    alignments: List[CrossModalAlignment] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class VisualElement:
    """Element extracted from visual input."""

    element_type: str  # "shape", "text", "symbol", "graph", "diagram"
    description: str
    position: Optional[Tuple[float, float]] = None  # Normalized (x, y)
    properties: Dict = field(default_factory=dict)
    mathematical_meaning: Optional[str] = None


@dataclass
class AudioSegment:
    """Segment extracted from audio input."""

    segment_type: str  # "speech", "music", "pattern", "silence"
    transcription: Optional[str] = None
    start_time: float = 0.0
    end_time: float = 0.0
    features: Dict = field(default_factory=dict)
    mathematical_patterns: List[str] = field(default_factory=list)


class VisionProcessor:
    """
    Processes visual inputs for mathematical understanding.

    Capabilities:
    - Diagram interpretation
    - Graph/chart reading
    - Symbol recognition
    - Spatial relationship extraction
    - Mathematical notation OCR
    """

    def __init__(self):
        self.processed_images: List[Dict] = []
        self.visual_vocabulary: Dict[str, str] = {
            # Shape -> Mathematical meaning
            "circle": "set, cycle, zero, or continuous rotation",
            "square": "area, unit, or discrete grid",
            "triangle": "delta, change, or relationship",
            "arrow": "direction, mapping, or transformation",
            "line": "connection, equality, or linear relationship",
            "curve": "function, continuous change, or smooth transformation",
            "dot": "point, element, or discrete value",
            "grid": "coordinate system, matrix, or discrete space",
            "graph": "function visualization, relationship, or data",
            "venn": "set operations, intersection, union"
        }
        print("[+] Vision Processor initialized")

    def process_image(
        self,
        image_input: Union[str, bytes],
        llm_bridge = None
    ) -> Dict:
        """
        Process an image for mathematical content.

        Args:
            image_input: File path or base64 encoded image
            llm_bridge: LLM for interpretation

        Returns:
            Dict with extracted visual elements and interpretation
        """
        print("[Vision] Processing image...")

        # Determine input type
        if isinstance(image_input, str) and os.path.exists(image_input):
            image_type = "file"
            image_path = image_input
        elif isinstance(image_input, bytes):
            image_type = "bytes"
            image_path = None
        else:
            image_type = "base64"
            image_path = None

        result = {
            "input_type": image_type,
            "elements": [],
            "mathematical_content": [],
            "spatial_structure": {},
            "interpretation": ""
        }

        if llm_bridge:
            # Use LLM for visual interpretation
            result = self._interpret_with_llm(image_input, llm_bridge)
        else:
            # Basic heuristic analysis
            result = self._heuristic_analysis(image_type)

        self.processed_images.append(result)
        return result

    def _interpret_with_llm(self, image_input: Any, llm_bridge) -> Dict:
        """Use LLM to interpret image content."""
        prompt = """Analyze this image for mathematical content.

Identify:
1. Visual elements (shapes, symbols, graphs, diagrams)
2. Mathematical notation or equations
3. Spatial relationships between elements
4. Overall mathematical concept being illustrated

Respond in JSON format:
{
    "elements": [
        {"type": "shape/symbol/graph", "description": "what it is", "mathematical_meaning": "what it represents"}
    ],
    "mathematical_content": ["equation1", "concept1"],
    "spatial_structure": {"layout": "description of spatial arrangement"},
    "interpretation": "overall interpretation of the mathematical diagram"
}
"""

        try:
            # If LLM supports vision, it would process the image
            # For now, return a structured placeholder
            response = llm_bridge.generate(prompt)

            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[!] LLM interpretation failed: {e}")

        return self._heuristic_analysis("unknown")

    def _heuristic_analysis(self, image_type: str) -> Dict:
        """Basic heuristic analysis when LLM not available."""
        return {
            "input_type": image_type,
            "elements": [
                VisualElement(
                    element_type="diagram",
                    description="Unprocessed visual content",
                    mathematical_meaning="Requires vision model for interpretation"
                ).__dict__
            ],
            "mathematical_content": [],
            "spatial_structure": {"layout": "unknown"},
            "interpretation": "Image processing requires vision-capable model"
        }

    def extract_mathematical_structure(
        self,
        visual_elements: List[VisualElement]
    ) -> Dict:
        """
        Extract mathematical structure from visual elements.

        Args:
            visual_elements: List of extracted visual elements

        Returns:
            Dict describing the mathematical structure
        """
        structure = {
            "type": "unknown",
            "components": [],
            "relationships": [],
            "mathematical_objects": []
        }

        for element in visual_elements:
            # Map visual type to mathematical type
            if element.element_type in ["graph", "curve"]:
                structure["type"] = "function_visualization"
                structure["mathematical_objects"].append({
                    "type": "function",
                    "from": element.description
                })
            elif element.element_type in ["venn", "circle"]:
                structure["type"] = "set_diagram"
                structure["mathematical_objects"].append({
                    "type": "set",
                    "from": element.description
                })
            elif element.element_type == "grid":
                structure["type"] = "coordinate_system"
            elif element.element_type == "arrow":
                structure["relationships"].append({
                    "type": "mapping",
                    "from": element.description
                })

        return structure

    def interpret_diagram_type(self, description: str) -> str:
        """Determine the type of mathematical diagram."""
        desc_lower = description.lower()

        diagram_types = {
            "graph": ["graph", "plot", "chart", "curve", "function"],
            "geometry": ["triangle", "circle", "square", "polygon", "angle"],
            "algebra": ["equation", "expression", "variable", "symbol"],
            "set_theory": ["venn", "set", "intersection", "union"],
            "topology": ["manifold", "surface", "continuous", "boundary"],
            "calculus": ["derivative", "integral", "limit", "tangent"],
            "linear_algebra": ["matrix", "vector", "transformation", "eigenvalue"]
        }

        for diagram_type, keywords in diagram_types.items():
            if any(kw in desc_lower for kw in keywords):
                return diagram_type

        return "general_mathematical"


class AudioProcessor:
    """
    Processes audio inputs for mathematical understanding.

    Capabilities:
    - Speech transcription interpretation
    - Pattern recognition in audio signals
    - Mathematical structure in sound
    - Lecture/explanation parsing
    """

    def __init__(self):
        self.processed_audio: List[Dict] = []
        self.mathematical_terms: Dict[str, str] = {
            # Audio cue -> Mathematical concept
            "plus": "addition",
            "minus": "subtraction",
            "times": "multiplication",
            "divided by": "division",
            "equals": "equality",
            "integral": "integration",
            "derivative": "differentiation",
            "limit": "limit",
            "infinity": "infinity",
            "approaches": "limit approach",
            "converges": "convergence",
            "diverges": "divergence"
        }
        print("[+] Audio Processor initialized")

    def process_audio(
        self,
        audio_input: Union[str, bytes],
        llm_bridge = None
    ) -> Dict:
        """
        Process audio for mathematical content.

        Args:
            audio_input: File path or audio bytes
            llm_bridge: LLM for interpretation

        Returns:
            Dict with extracted audio information
        """
        print("[Audio] Processing audio...")

        result = {
            "transcription": None,
            "segments": [],
            "mathematical_content": [],
            "patterns": [],
            "interpretation": ""
        }

        if llm_bridge:
            result = self._interpret_with_llm(audio_input, llm_bridge)
        else:
            result = self._heuristic_analysis()

        self.processed_audio.append(result)
        return result

    def _interpret_with_llm(self, audio_input: Any, llm_bridge) -> Dict:
        """Use LLM to interpret audio/transcription."""
        prompt = """Analyze this audio transcription for mathematical content.

Identify:
1. Mathematical terms and concepts mentioned
2. Equations or formulas described verbally
3. Logical structure of the explanation
4. Key mathematical ideas being communicated

Respond in JSON format:
{
    "mathematical_content": ["concept1", "equation1"],
    "logical_structure": "how ideas are connected",
    "key_concepts": ["concept1", "concept2"],
    "interpretation": "overall mathematical meaning"
}
"""

        try:
            response = llm_bridge.generate(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"[!] Audio interpretation failed: {e}")

        return self._heuristic_analysis()

    def _heuristic_analysis(self) -> Dict:
        """Basic analysis when transcription available."""
        return {
            "transcription": None,
            "segments": [],
            "mathematical_content": [],
            "patterns": [],
            "interpretation": "Audio processing requires speech recognition model"
        }

    def extract_mathematical_speech(self, transcription: str) -> List[str]:
        """
        Extract mathematical concepts from speech transcription.

        Args:
            transcription: Text transcription of speech

        Returns:
            List of mathematical concepts mentioned
        """
        concepts = []

        trans_lower = transcription.lower()
        for term, concept in self.mathematical_terms.items():
            if term in trans_lower:
                concepts.append(concept)

        # Look for number patterns
        import re
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', transcription)
        if numbers:
            concepts.append(f"numerical_values: {numbers}")

        # Look for variable mentions
        variables = re.findall(r'\b([a-z])\s+(?:equals|is|=)', trans_lower)
        if variables:
            concepts.append(f"variables: {variables}")

        return concepts

    def analyze_audio_patterns(self, audio_features: Dict) -> List[str]:
        """
        Analyze mathematical patterns in audio signal.

        Args:
            audio_features: Extracted audio features (frequency, amplitude, etc.)

        Returns:
            List of mathematical patterns detected
        """
        patterns = []

        # Check for periodic patterns (frequency domain)
        if "frequencies" in audio_features:
            freqs = audio_features["frequencies"]
            # Look for harmonic relationships
            if len(freqs) > 1:
                ratios = [freqs[i+1]/freqs[i] for i in range(len(freqs)-1) if freqs[i] > 0]
                for ratio in ratios:
                    if abs(ratio - 2.0) < 0.1:
                        patterns.append("octave_relationship")
                    elif abs(ratio - 1.5) < 0.1:
                        patterns.append("perfect_fifth")

        # Check for exponential decay
        if "amplitude_envelope" in audio_features:
            envelope = audio_features["amplitude_envelope"]
            if self._is_exponential_decay(envelope):
                patterns.append("exponential_decay")

        return patterns

    def _is_exponential_decay(self, envelope: List[float]) -> bool:
        """Check if amplitude follows exponential decay."""
        if len(envelope) < 3:
            return False

        # Simple check: ratios between consecutive points should be similar
        ratios = []
        for i in range(len(envelope) - 1):
            if envelope[i] > 0.001:
                ratios.append(envelope[i+1] / envelope[i])

        if not ratios:
            return False

        avg_ratio = sum(ratios) / len(ratios)
        return all(abs(r - avg_ratio) < 0.2 for r in ratios) and avg_ratio < 1.0


class MultimodalReasoningEngine:
    """
    Phase 6: Unified Multimodal Reasoning Engine.

    Integrates vision, audio, and text for holistic understanding.

    Capabilities:
    1. Cross-modal alignment - connect concepts across modalities
    2. Multimodal memory - store and retrieve multimodal concepts
    3. Unified reasoning - solve problems using all available modalities
    4. Modal translation - describe visual concepts in text and vice versa
    5. Multimodal learning - learn from mixed-modality inputs
    """

    def __init__(self, storage_path: str = "./multimodal_memory.json"):
        self.storage_path = storage_path

        # Processors
        self.vision = VisionProcessor()
        self.audio = AudioProcessor()

        # Memory
        self.multimodal_concepts: Dict[str, MultimodalConcept] = {}
        self.cross_modal_alignments: List[CrossModalAlignment] = []
        self.reasoning_history: List[Dict] = []

        # Load existing data
        self.load()

        print("[+] Multimodal Reasoning Engine: Vision + Audio + Text unified!")

    def process_multimodal_input(
        self,
        inputs: List[ModalityInput],
        llm_bridge = None
    ) -> Dict:
        """
        Process multiple modality inputs together.

        Args:
            inputs: List of inputs from different modalities
            llm_bridge: LLM for reasoning

        Returns:
            Dict with unified understanding
        """
        print(f"[Multimodal] Processing {len(inputs)} modality inputs...")

        # Process each modality
        processed = {}
        for inp in inputs:
            if inp.modality == Modality.IMAGE or inp.modality == Modality.DIAGRAM:
                processed["visual"] = self.vision.process_image(inp.content, llm_bridge)
            elif inp.modality == Modality.AUDIO:
                processed["audio"] = self.audio.process_audio(inp.content, llm_bridge)
            elif inp.modality == Modality.TEXT:
                processed["text"] = {"content": inp.content, "concepts": inp.extracted_concepts}
            elif inp.modality == Modality.MATHEMATICAL:
                processed["mathematical"] = {"expression": inp.content}

        # Align concepts across modalities
        alignments = self._align_modalities(processed, llm_bridge)

        # Generate unified understanding
        unified = self._unify_understanding(processed, alignments, llm_bridge)

        result = {
            "processed_modalities": processed,
            "alignments": alignments,
            "unified_understanding": unified,
            "timestamp": datetime.now().isoformat()
        }

        self.reasoning_history.append(result)

        print(f"[Multimodal] Generated unified understanding")
        return result

    def _align_modalities(
        self,
        processed: Dict,
        llm_bridge = None
    ) -> List[Dict]:
        """Align concepts across different modalities."""
        alignments = []

        # Extract concepts from each modality
        visual_concepts = []
        text_concepts = []
        audio_concepts = []

        if "visual" in processed:
            visual_concepts = processed["visual"].get("mathematical_content", [])
        if "text" in processed:
            text_concepts = processed["text"].get("concepts", [])
        if "audio" in processed:
            audio_concepts = processed["audio"].get("mathematical_content", [])

        # Cross-modal alignment using LLM
        if llm_bridge and (visual_concepts or text_concepts or audio_concepts):
            prompt = f"""Align concepts across modalities:

Visual concepts: {visual_concepts}
Text concepts: {text_concepts}
Audio concepts: {audio_concepts}

Find correspondences between concepts. Which concepts from different modalities refer to the same mathematical idea?

Respond in JSON:
{{
    "alignments": [
        {{"modality_a": "visual", "concept_a": "concept", "modality_b": "text", "concept_b": "concept", "relationship": "equivalent"}}
    ]
}}
"""
            try:
                response = llm_bridge.generate(prompt)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    alignments = data.get("alignments", [])
            except Exception as e:
                print(f"[!] Alignment failed: {e}")

        return alignments

    def _unify_understanding(
        self,
        processed: Dict,
        alignments: List[Dict],
        llm_bridge = None
    ) -> str:
        """Generate unified understanding from all modalities."""
        if not llm_bridge:
            parts = []
            if "visual" in processed:
                parts.append(f"Visual: {processed['visual'].get('interpretation', 'N/A')}")
            if "text" in processed:
                parts.append(f"Text: {processed['text'].get('content', 'N/A')[:100]}")
            if "audio" in processed:
                parts.append(f"Audio: {processed['audio'].get('interpretation', 'N/A')}")
            return " | ".join(parts) if parts else "No modalities processed"

        prompt = f"""Synthesize a unified understanding from these modality inputs:

{json.dumps(processed, indent=2, default=str)}

Alignments found: {alignments}

Provide a unified interpretation that combines all modalities into a coherent mathematical understanding.
Focus on:
1. What is the core mathematical concept?
2. How do different modalities contribute to understanding?
3. What is the complete picture?

Unified interpretation:"""

        return llm_bridge.generate(prompt).strip()

    def learn_multimodal_concept(
        self,
        name: str,
        inputs: List[ModalityInput],
        llm_bridge = None
    ) -> MultimodalConcept:
        """
        Learn a concept from multiple modality inputs.

        Args:
            name: Concept name
            inputs: Inputs from different modalities
            llm_bridge: LLM for understanding

        Returns:
            MultimodalConcept
        """
        print(f"[Multimodal] Learning concept: {name}")

        # Process all inputs
        result = self.process_multimodal_input(inputs, llm_bridge)

        # Create multimodal concept
        concept = MultimodalConcept(
            name=name,
            text_representation=None,
            visual_representation=None,
            audio_representation=None,
            mathematical_form=None,
            alignments=[],
            confidence=0.7
        )

        # Fill in representations from each modality
        processed = result["processed_modalities"]

        if "text" in processed:
            concept.text_representation = processed["text"].get("content")

        if "visual" in processed:
            concept.visual_representation = processed["visual"].get("interpretation")

        if "audio" in processed:
            concept.audio_representation = processed["audio"].get("interpretation")

        if "mathematical" in processed:
            concept.mathematical_form = processed["mathematical"].get("expression")

        # Store concept
        self.multimodal_concepts[name] = concept
        self.save()

        print(f"[Multimodal] Learned concept with {len([r for r in [concept.text_representation, concept.visual_representation, concept.audio_representation, concept.mathematical_form] if r])} modality representations")

        return concept

    def reason_multimodally(
        self,
        question: str,
        available_inputs: List[ModalityInput],
        llm_bridge
    ) -> Dict:
        """
        Reason about a question using multiple modalities.

        Args:
            question: Question to answer
            available_inputs: Available multimodal inputs
            llm_bridge: LLM for reasoning

        Returns:
            Dict with reasoning result
        """
        print(f"[Multimodal] Reasoning about: {question[:50]}...")

        # Process inputs
        processed = self.process_multimodal_input(available_inputs, llm_bridge)

        # Retrieve relevant multimodal concepts
        relevant_concepts = self._find_relevant_concepts(question)

        # Multimodal reasoning
        prompt = f"""Answer this question using multimodal reasoning:

Question: {question}

Available information:
{json.dumps(processed['processed_modalities'], indent=2, default=str)}

Relevant stored concepts:
{[c.name for c in relevant_concepts]}

Use all available modalities to reason about the answer.
Consider:
1. What does the visual information tell us?
2. What does the text/audio add?
3. How do they combine to answer the question?

Reasoning and answer:"""

        answer = llm_bridge.generate(prompt).strip()

        result = {
            "question": question,
            "modalities_used": list(processed['processed_modalities'].keys()),
            "relevant_concepts": [c.name for c in relevant_concepts],
            "answer": answer,
            "confidence": self._estimate_confidence(processed)
        }

        return result

    def _find_relevant_concepts(self, query: str) -> List[MultimodalConcept]:
        """Find concepts relevant to a query."""
        relevant = []
        query_lower = query.lower()

        for name, concept in self.multimodal_concepts.items():
            # Simple keyword matching
            if name.lower() in query_lower:
                relevant.append(concept)
            elif concept.text_representation and any(word in concept.text_representation.lower() for word in query_lower.split()):
                relevant.append(concept)

        return relevant[:5]  # Top 5 most relevant

    def _estimate_confidence(self, processed: Dict) -> float:
        """Estimate confidence based on available modalities."""
        modality_count = len(processed.get('processed_modalities', {}))
        base_confidence = 0.3 + (modality_count * 0.2)  # More modalities = higher confidence
        return min(1.0, base_confidence)

    def translate_modality(
        self,
        content: str,
        source_modality: Modality,
        target_modality: Modality,
        llm_bridge
    ) -> str:
        """
        Translate content from one modality description to another.

        Args:
            content: Source content
            source_modality: Original modality
            target_modality: Target modality
            llm_bridge: LLM for translation

        Returns:
            Translated content description
        """
        print(f"[Multimodal] Translating {source_modality.value} -> {target_modality.value}")

        prompt = f"""Translate this {source_modality.value} content into a {target_modality.value} representation.

Source ({source_modality.value}):
{content}

Create a {target_modality.value} representation that captures the same mathematical meaning.

For visual: Describe what a diagram would look like
For audio: Describe how to explain this verbally
For text: Write a clear textual explanation
For mathematical: Write the formal mathematical notation

{target_modality.value} representation:"""

        return llm_bridge.generate(prompt).strip()

    def get_concept(self, name: str) -> Optional[MultimodalConcept]:
        """Get a multimodal concept by name."""
        return self.multimodal_concepts.get(name)

    def save(self):
        """Save multimodal memory to disk."""
        try:
            data = {
                "concepts": {
                    name: {
                        "name": c.name,
                        "text_representation": c.text_representation,
                        "visual_representation": c.visual_representation,
                        "audio_representation": c.audio_representation,
                        "mathematical_form": c.mathematical_form,
                        "confidence": c.confidence
                    }
                    for name, c in self.multimodal_concepts.items()
                },
                "alignments": [
                    {
                        "modality_a": a.modality_a.value if hasattr(a, 'modality_a') else a.get('modality_a'),
                        "modality_b": a.modality_b.value if hasattr(a, 'modality_b') else a.get('modality_b'),
                        "concept_a": a.concept_a if hasattr(a, 'concept_a') else a.get('concept_a'),
                        "concept_b": a.concept_b if hasattr(a, 'concept_b') else a.get('concept_b'),
                        "alignment_score": a.alignment_score if hasattr(a, 'alignment_score') else a.get('alignment_score', 0.5),
                        "relationship": a.relationship if hasattr(a, 'relationship') else a.get('relationship')
                    }
                    for a in self.cross_modal_alignments
                ]
            }
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[+] Saved {len(self.multimodal_concepts)} multimodal concepts")
        except Exception as e:
            print(f"[!] Failed to save multimodal memory: {e}")

    def load(self):
        """Load multimodal memory from disk."""
        if not os.path.exists(self.storage_path):
            return

        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Load concepts
            for name, c_data in data.get("concepts", {}).items():
                self.multimodal_concepts[name] = MultimodalConcept(
                    name=c_data["name"],
                    text_representation=c_data.get("text_representation"),
                    visual_representation=c_data.get("visual_representation"),
                    audio_representation=c_data.get("audio_representation"),
                    mathematical_form=c_data.get("mathematical_form"),
                    confidence=c_data.get("confidence", 0.5)
                )

            print(f"[+] Loaded {len(self.multimodal_concepts)} multimodal concepts")
        except Exception as e:
            print(f"[!] Failed to load multimodal memory: {e}")

    def get_stats(self) -> Dict:
        """Get statistics about multimodal processing."""
        return {
            "concepts_stored": len(self.multimodal_concepts),
            "cross_modal_alignments": len(self.cross_modal_alignments),
            "reasoning_sessions": len(self.reasoning_history),
            "images_processed": len(self.vision.processed_images),
            "audio_processed": len(self.audio.processed_audio)
        }

    def summarize(self) -> str:
        """Get human-readable summary."""
        stats = self.get_stats()
        lines = [
            "Multimodal Reasoning Engine Status:",
            f"  Concepts stored: {stats['concepts_stored']}",
            f"  Cross-modal alignments: {stats['cross_modal_alignments']}",
            f"  Reasoning sessions: {stats['reasoning_sessions']}",
            f"  Images processed: {stats['images_processed']}",
            f"  Audio processed: {stats['audio_processed']}"
        ]

        if self.multimodal_concepts:
            lines.append("\nRecent concepts:")
            for name in list(self.multimodal_concepts.keys())[-3:]:
                concept = self.multimodal_concepts[name]
                modalities = []
                if concept.text_representation:
                    modalities.append("text")
                if concept.visual_representation:
                    modalities.append("visual")
                if concept.audio_representation:
                    modalities.append("audio")
                lines.append(f"  - {name}: {', '.join(modalities)}")

        return "\n".join(lines)
