import Mathlib.Data.Set.Lattice
import Mathlib.Data.Set.Function
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import LLMlean
import Lean

def printEnv : IO Unit := do
  -- Basic environment variable check
  IO.println "=== LLMLean Environment Variables ==="
  let api ← IO.getEnv "LLMLEAN_API"
  let model ← IO.getEnv "LLMLEAN_MODEL"
  let endpoint ← IO.getEnv "LLMLEAN_ENDPOINT"
  let prompt ← IO.getEnv "LLMLEAN_PROMPT"
  let samples ← IO.getEnv "LLMLEAN_NUMSAMPLES"

  IO.println s!"LLMLEAN_API raw value: {api}"
  IO.println s!"LLMLEAN_API with default: {api.getD "openai"}"
  IO.println s!"API selection would be: {
    match api.getD "openai" with
    | "ollama" => "Ollama API"
    | "together" => "Together API"
    | "deepseek" => "DeepSeek API"
    | "claude" => "Claude API"
    | "anthropic" => "Claude API"
    | _ => "OpenAI API"
  }"

  IO.println "\n=== Other Variables ==="
  IO.println s!"LLMLEAN_MODEL: {model}"
  IO.println s!"LLMLEAN_ENDPOINT: {endpoint}"
  IO.println s!"LLMLEAN_PROMPT: {prompt}"
  IO.println s!"LLMLEAN_NUMSAMPLES: {samples}"

#eval printEnv

section

variable {α β : Type*}
variable (f : α → β)
variable (s t : Set α)
variable (u v : Set β)

open Function
open Set

example : f ⁻¹' (u ∩ v) = f ⁻¹' u ∩ f ⁻¹' v := by
  ext x
  constructor
  · intro h
    rw [mem_preimage] at h
    rw [mem_inter_iff]
    rw [mem_preimage]
    rw [mem_preimage]
    exact ⟨h.1, h.2⟩
  · intro h
    rw [mem_inter_iff] at h
    rw [mem_preimage]
    rw [mem_preimage] at h
    rw [mem_preimage] at h
    rw [mem_inter_iff]
    exact ⟨h.1, h.2⟩
  -- llmstep "rw "



example : f '' (s ∪ t) = f '' s ∪ f '' t := by
  -- llmqed
  -- llmstep "rw "
  sorry

example : s ⊆ f ⁻¹' (f '' s) := by
  -- llmqed
  sorry

example : f '' s ⊆ v ↔ s ⊆ f ⁻¹' v := by
  sorry

example (h : Injective f) : f ⁻¹' (f '' s) ⊆ s := by
  sorry

example : f '' (f ⁻¹' u) ⊆ u := by
  sorry

example (h : Surjective f) : u ⊆ f '' (f ⁻¹' u) := by
  sorry

example (h : s ⊆ t) : f '' s ⊆ f '' t := by
  sorry

example (h : u ⊆ v) : f ⁻¹' u ⊆ f ⁻¹' v := by
  sorry

example : f ⁻¹' (u ∪ v) = f ⁻¹' u ∪ f ⁻¹' v := by
  sorry

example : f '' (s ∩ t) ⊆ f '' s ∩ f '' t := by
  sorry

example (h : Injective f) : f '' s ∩ f '' t ⊆ f '' (s ∩ t) := by
  sorry

example : f '' s \ f '' t ⊆ f '' (s \ t) := by
  sorry

example : f ⁻¹' u \ f ⁻¹' v ⊆ f ⁻¹' (u \ v) := by
  sorry

example : f '' s ∩ v = f '' (s ∩ f ⁻¹' v) := by
  sorry

example : f '' (s ∩ f ⁻¹' u) ⊆ f '' s ∩ u := by
  sorry

example : s ∩ f ⁻¹' u ⊆ f ⁻¹' (f '' s ∩ u) := by
  sorry

example : s ∪ f ⁻¹' u ⊆ f ⁻¹' (f '' s ∪ u) := by
  sorry

variable {I : Type*} (A : I → Set α) (B : I → Set β)

example : (f '' ⋃ i, A i) = ⋃ i, f '' A i := by
  sorry

example : (f '' ⋂ i, A i) ⊆ ⋂ i, f '' A i := by
  sorry

example (i : I) (injf : Injective f) : (⋂ i, f '' A i) ⊆ f '' ⋂ i, A i := by
  sorry

example : (f ⁻¹' ⋃ i, B i) = ⋃ i, f ⁻¹' B i := by
  sorry

example : (f ⁻¹' ⋂ i, B i) = ⋂ i, f ⁻¹' B i := by
  sorry

example : InjOn f s ↔ ∀ x₁ ∈ s, ∀ x₂ ∈ s, f x₁ = f x₂ → x₁ = x₂ :=
  Iff.refl _

end

section

open Set Real

example : InjOn log { x | x > 0 } := by
  sorry

example : range exp = { y | y > 0 } := by
  sorry

example : InjOn sqrt { x | x ≥ 0 } := by
  sorry

example : InjOn (fun x ↦ x ^ 2) { x : ℝ | x ≥ 0 } := by
  sorry

example : sqrt '' { x | x ≥ 0 } = { y | y ≥ 0 } := by
  sorry

example : (range fun x ↦ x ^ 2) = { y : ℝ | y ≥ 0 } := by
  sorry

end

section
variable {α β : Type*} [Inhabited α]

#check (default : α)

variable (P : α → Prop) (h : ∃ x, P x)

#check Classical.choose h

example : P (Classical.choose h) :=
  Classical.choose_spec h

noncomputable section

open Classical

def inverse (f : α → β) : β → α := fun y : β ↦
  if h : ∃ x, f x = y then Classical.choose h else default

theorem inverse_spec {f : α → β} (y : β) (h : ∃ x, f x = y) : f (inverse f y) = y := by
  rw [inverse, dif_pos h]
  exact Classical.choose_spec h

variable (f : α → β)

open Function

example : Injective f ↔ LeftInverse (inverse f) f := by
  sorry

example : Surjective f ↔ RightInverse (inverse f) f :=
  sorry

end

section
variable {α : Type*}
open Function

theorem Cantor : ∀ f : α → Set α, ¬Surjective f := by
  intro f surjf
  let S := { i | i ∉ f i }
  rcases surjf S with ⟨j, h⟩
  have h₁ : j ∉ f j := by
    intro h'
    have : j ∉ f j := by rwa [h] at h'
    contradiction
  have h₂ : j ∈ S := by sorry
  have h₃ : j ∉ S := by sorry
  contradiction

-- COMMENTS: TODO: improve this
end
