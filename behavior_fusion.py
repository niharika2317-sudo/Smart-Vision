class BehaviorFusion:

    def fuse(self, movement, pose, weapon_detected):

        if weapon_detected and movement in ("Running", "Aggressive Movement"):
            return "Possible Attack"

        if weapon_detected:
            return "Armed Person"

        if pose == "Possible Fall":
            return "Fall Detected"

        if pose == "Raised Hand":
            return "Raised Hand"

        return movement
