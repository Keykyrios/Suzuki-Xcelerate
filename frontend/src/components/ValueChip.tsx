import type { ReactNode } from "react";
import type { EvidenceLabel } from "../types/api";

const LABEL_STYLE: Record<EvidenceLabel, string> = {
  OBSERVED: "label-observed",
  PUBLIC_REFERENCE: "label-reference",
  CONFIGURED: "label-configured",
  SYNTHETIC: "label-synthetic",
  RECONSTRUCTED: "label-reconstructed",
  SCENARIO_ASSUMPTION: "label-synthetic",
  MODEL_ESTIMATE: "label-estimate",
  SIMULATED: "label-estimate",
  UNCALIBRATED_CONCEPT_TARGET: "label-target",
} as Record<EvidenceLabel, string>;

export function ValueChip({
  label,
  unit,
  evidence,
  children,
}: {
  label: string;
  unit: string;
  evidence: EvidenceLabel;
  children: ReactNode;
}) {
  return (
    <div className="value-chip" data-testid={`value-${label}`}>
      <span className="value-label">{label}</span>
      <span className="value-number">
        {children} <span className="value-unit">{unit}</span>
      </span>
      <span className={`value-evidence ${LABEL_STYLE[evidence] ?? "label-synthetic"}`}>
        {evidence.replaceAll("_", " ")}
      </span>
    </div>
  );
}
