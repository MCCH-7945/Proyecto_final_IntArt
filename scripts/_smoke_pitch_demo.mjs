/** Smoke test: pitch bounds + feature map logic from the web demo. */
import { readFileSync } from "fs";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "web_demo/keeper_probability_demo.html"), "utf8");
const model = JSON.parse(readFileSync(join(root, "web_demo/save_probability_model.json"), "utf8"));

const GOAL_WIDTH_M = 7.32;
const GOAL_HEIGHT_M = 2.44;
const PITCH_RATIOS = {
  penaltyDepth: 16.5 / GOAL_HEIGHT_M,
  penaltyWidth: 40.32 / GOAL_WIDTH_M,
  goalAreaDepth: 5.5 / GOAL_HEIGHT_M,
  goalAreaWidth: 18.32 / GOAL_WIDTH_M,
};
const PENALTY_U_HALF = (PITCH_RATIOS.penaltyWidth - 1) / 2;
const PITCH_BOUNDS = {
  uMin: 0.5 - PENALTY_U_HALF,
  uMax: 0.5 + PENALTY_U_HALF,
  vMin: -PITCH_RATIOS.penaltyDepth,
  vMax: 0,
};
const GOAL_AREA_U_HALF = (PITCH_RATIOS.goalAreaWidth - 1) / 2;
const GOAL_AREA_BOUNDS = {
  uMin: 0.5 - GOAL_AREA_U_HALF,
  uMax: 0.5 + GOAL_AREA_U_HALF,
  vMin: -PITCH_RATIOS.goalAreaDepth,
  vMax: 0,
};

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}
function clampToPitchArea(u, v) {
  return { u: clamp(u, PITCH_BOUNDS.uMin, PITCH_BOUNDS.uMax), v: clamp(v, PITCH_BOUNDS.vMin, PITCH_BOUNDS.vMax) };
}
function inPitch(u, v) {
  return u >= PITCH_BOUNDS.uMin && u <= PITCH_BOUNDS.uMax && v >= PITCH_BOUNDS.vMin && v <= PITCH_BOUNDS.vMax;
}
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
function binaryEntropyBits(p) {
  const q = clamp(p, 1e-9, 1 - 1e-9);
  return -(q * Math.log2(q) + (1 - q) * Math.log2(1 - q));
}
function mlProbability(state, u, v, model) {
  const lateral = (u - state.keeperU) * GOAL_WIDTH_M;
  const vertical = (v - state.keeperV) * GOAL_HEIGHT_M;
  const features = {
    keeper_center_u: state.keeperU,
    keeper_center_v: state.keeperV,
    keeper_body_width: Math.max(0.35, state.reachX * 0.66),
    keeper_body_height: Math.max(1.0, state.reachZ * 1.62),
    keeper_hand_span: Math.max(0.9, state.reachX * 1.55),
    keeper_foot_span: Math.max(0.3, state.reachX * 0.58),
    keeper_polygon_area_uv: 0.12,
    keeper_pose_confidence: 0.72,
    ball_position_u: state.ballU,
    ball_position_v: state.ballV,
    goal_entry_u: u,
    goal_entry_v: v,
    shot_ball_speed: state.shotSpeed,
    time_ball_to_goal: state.ballTime,
    reaction_time: state.reaction,
    lateral_delta_m: lateral,
    vertical_delta_m: vertical,
    distance_keeper_to_target_m: Math.hypot(lateral, vertical),
    distance_ball_to_target_m: Math.hypot((u - state.ballU) * GOAL_WIDTH_M, (v - state.ballV) * GOAL_HEIGHT_M),
    distance_keeper_to_ball_m: Math.hypot((state.ballU - state.keeperU) * GOAL_WIDTH_M, (state.ballV - state.keeperV) * GOAL_HEIGHT_M),
    keeper_outside_goal: Math.max(0, Math.abs(state.keeperU - 0.5) - 0.5) * 2,
  };
  let logit = model.intercept;
  for (let i = 0; i < model.feature_columns.length; i += 1) {
    const name = model.feature_columns[i];
    let value = Number(features[name]);
    if (!Number.isFinite(value)) value = model.imputer_median[i];
    const scaled = (value - model.scaler_mean[i]) / Math.max(1e-9, model.scaler_scale[i]);
    logit += model.coef[i] * scaled;
  }
  return sigmoid(logit);
}

const checks = [];
checks.push(["html_has_clampToPitchArea", html.includes("function clampToPitchArea")]);
checks.push(["html_has_drawPitchPlayableZone", html.includes("function drawPitchPlayableZone")]);
checks.push(["html_has_monte_carlo", html.includes("function computeDecisionAnalysis") && html.includes("MC_SAMPLE_COUNT")]);
checks.push(["html_has_entropy", html.includes("function binaryEntropyBits") && html.includes("entropyBits")]);
checks.push(["html_has_keeper_optimization", html.includes("function optimizeKeeperPosition") && html.includes("applyBestKeeperButton")]);
checks.push(["html_has_analysis_visuals", html.includes("mcHistogram") && html.includes("currentExpectedBar")]);
checks.push(["html_has_fixed_field_zoom", html.includes("FIXED_VIEW_ZOOM") && !html.includes("id=\"viewZoom\"")]);
checks.push(["html_has_interpretability_panel", html.includes("interpretationPanel") && html.includes("mlFeatureContributions")]);
checks.push(["html_has_dropdown_guide", html.includes("<details class=\"guide-details\">") && html.includes("<summary>Guía de barras</summary>")]);
checks.push(["html_has_alpha_selector", html.includes("data-alpha=\"0.01\"") && html.includes("ciAlpha") && html.includes("monte_carlo_alpha")]);
checks.push(["html_has_ci_width", html.includes("mcIntervalWidth") && html.includes("summary.upper - summary.lower")]);
checks.push(["html_uncertainty_does_not_mix_probability", !html.includes("uncertaintyMix")]);
checks.push(["model_feature_count", model.feature_columns.length >= 21]);

const state = { keeperU: 0.5, keeperV: -0.55, reachX: 1.18, reachZ: 0.95, ballU: 0.52, ballV: -2.35, shotSpeed: 82, ballTime: 0.55, reaction: 0.32 };
const clipped = clampToPitchArea(3.5, 0.8);
checks.push(["clamp_u_high", clipped.u === PITCH_BOUNDS.uMax]);
checks.push(["clamp_v_high", clipped.v === PITCH_BOUNDS.vMax]);
const corner = clampToPitchArea(PITCH_BOUNDS.uMin, PITCH_BOUNDS.vMin);
checks.push(["corner_in_pitch", inPitch(corner.u, corner.v)]);
const small = clampToPitchArea(0.5, -1.0);
checks.push(["small_box_v", small.v >= GOAL_AREA_BOUNDS.vMin && small.v <= GOAL_AREA_BOUNDS.vMax]);
const p = mlProbability(state, 0.74, 0.58, model);
checks.push(["ml_finite", Number.isFinite(p) && p > 0 && p < 1]);
checks.push(["entropy_max_at_half", Math.abs(binaryEntropyBits(0.5) - 1) < 1e-9]);
checks.push(["entropy_low_at_edge", binaryEntropyBits(0.01) < 0.1]);

let failed = 0;
for (const [name, ok] of checks) {
  console.log(`${ok ? "OK" : "FAIL"} ${name}`);
  if (!ok) failed += 1;
}
process.exit(failed ? 1 : 0);
