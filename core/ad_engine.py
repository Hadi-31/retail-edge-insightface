import yaml
import time

class AdEngine:
    def __init__(self, cfg_path):
        with open(cfg_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self.state = {}  # cooldown dict: id -> last_time

    def in_cooldown(self, pid, cooldown):
        t = self.state.get(pid, 0)
        return (time.time() - t) < cooldown

    def mark(self, pid):
        self.state[pid] = time.time()

    def choose(self, context):
        '''
        context: dict with keys ->
            people_count, time_of_day, persons: [ {id, age, gender, expression, clothing_style, is_child}, ...]
        returns: (ad_path, reason_str)
        '''
        g = self.cfg.get('global', {})
        rules = self.cfg.get('rules', [])
        guard = self.cfg.get('guardrails', {})
        cooldown = g.get('cooldown_seconds', 45)

        people = context.get('persons', [])
        # Default: no ad if only children
        if guard.get('ignore_children', True):
            if all(p.get('is_child', False) for p in people) and people:
                return (None, "Guardrail: children-only scene")

        # Build summary persona for the 'dominant' person (first non-child)
        dominant = None
        for p in people:
            if not p.get('is_child', False):
                dominant = p
                break
        if dominant is None and people:
            dominant = people[0]

        # Evaluate rules in order
        for rule in rules:
            cond = rule.get('when', {})
            if not self._match(cond, context, dominant):
                continue
            ad = rule.get('show')
            if dominant and self.in_cooldown(dominant['id'], cooldown):
                continue
            if dominant:
                self.mark(dominant['id'])
            reason = f"Matched rule '{rule.get('name')}'"
            return (ad, reason)

        return (None, "No rule matched")

    def _match(self, cond, ctx, dom):
        # people_count operators
        pc = ctx.get('people_count', 0)
        if 'people_count' in cond:
            expr = cond['people_count']
            if expr.startswith(">="):
                if not (pc >= int(expr[2:])): return False
            elif expr.startswith("=="):
                if not (pc == int(expr[2:])): return False
            elif expr.startswith("<="):
                if not (pc <= int(expr[2:])): return False

        # time of day
        if 'time_of_day_any_of' in cond:
            if ctx.get('time_of_day') not in cond['time_of_day_any_of']:
                return False

        if dom is None:
            return True

        # age range
        if 'age_range' in cond:
            lo, hi = cond['age_range']
            age = dom.get('age')
            if age is None or not (lo <= age <= hi):
                return False

        # expression
        if 'expression_any_of' in cond:
            if dom.get('expression') not in cond['expression_any_of']:
                return False

        # clothing style
        if 'clothing_style_any_of' in cond:
            if dom.get('clothing_style') not in cond['clothing_style_any_of']:
                return False

        return True
