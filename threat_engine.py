import config
from utils import point_in_box


class ThreatEngine:
    BEHAVIOR_SCORES = {"Running": 35, "Raised Hand": 30, "Fall Detected": 15,
                       "Aggressive Movement": 65, "Armed Person": 60, "Possible Attack": 90}

    def evaluate(self, persons, weapons, behaviors):
        def touches_person(weapon):
            x1, y1, x2, y2 = map(float, weapon["box"])
            return any(x1 <= float(p["box"][2]) and x2 >= float(p["box"][0])
                       and y1 <= float(p["box"][3]) and y2 >= float(p["box"][1])
                       for p in persons)
        held = [touches_person(w) for w in weapons]
        weapon_score = min(100, max((40 + 40 * float(w["confidence"]) + (20 if h else 0)
                                     for w, h in zip(weapons, held)), default=0))
        behavior_score = max((self.BEHAVIOR_SCORES.get(b, 0) for b in behaviors), default=0)
        reasons = []
        for w, h in zip(weapons, held):
            reasons.append("Weapon model candidate · review required")
            if h: reasons.append("Candidate overlaps a person")
        if weapons and not any(held):
            reasons.append("Weapon candidate without nearby person")
        for b in behaviors:
            if b in self.BEHAVIOR_SCORES: reasons.append(b)
        if sum(b in ("Running", "Aggressive Movement") for b in behaviors) >= 3:
            behavior_score = min(100, behavior_score + 15)
            reasons.append("Crowd Panic")
        score = round(0.6 * max(weapon_score, behavior_score) + 0.4 * min(weapon_score, behavior_score))
        level = "HIGH" if score >= config.SCORE_HIGH else "MEDIUM" if score >= config.SCORE_LOW else "LOW"
        return {"score": score, "level": level, "weapon_score": round(weapon_score),
                "behavior_score": round(behavior_score), "reasons": list(dict.fromkeys(reasons))}
