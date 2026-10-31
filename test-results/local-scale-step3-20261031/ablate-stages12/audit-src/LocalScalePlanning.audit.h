#ifndef HECATE_SUPPORT_LOCALSCALEPLANNING_H
#define HECATE_SUPPORT_LOCALSCALEPLANNING_H

#include "hecate/Support/BootstrapInputAlignment.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <tuple>

namespace hecate {
inline double auditUnitFloor(double waterline) {
  const char *s = std::getenv("AUDIT_UNIT_FLOOR");
  return s ? std::strtod(s, nullptr) : waterline;
}


// Levels count Q consumed from anchor. Flex describes a private arithmetic
// tree whose coefficient encoding can shift its scale without changing values.
struct LocalScaleValue {
  int64_t level = 0;
  double scale = 0;
  int64_t nominal = 0;
  double minShift = 1, maxShift = 0;
  unsigned edits = 0;
  bool flexible() const { return edits && minShift <= maxShift; }
};

enum class LocalScaleKind { MulCP, MulCC, AddCC, AddCP, Unary };
struct LocalScaleAction {
  double shift = 0;
  unsigned modswitch = 0, rescales = 0;
  bool retune = false, unit = false;
};
struct LocalScalePlan {
  LocalScaleValue result;
  std::array<LocalScaleAction, 2> inputs;
  double encodingScale = 0;
  unsigned resultRescales = 0;
  int flexOperand = -1;
  unsigned work = 0;
};

class LocalScalePlanning {
public:
  static constexpr unsigned maxEdits = 8;
  static constexpr double equalScale = 1e-10;
  LocalScalePlanning(const ModulusChain &chain, int64_t anchor,
                     double waterline, int64_t budget, unsigned bootLevel,
                     double bootScale)
      : chain(chain), anchor(anchor), waterline(waterline), budget(budget),
        bootLevel(bootLevel), bootScale(bootScale) {}

  std::optional<LocalScalePlan>
  choose(LocalScaleKind kind, LocalScaleValue a, LocalScaleValue b,
         bool editable, bool optimize,
         const std::array<int64_t, 3> &forecast) const {
    if (!valid(a) ||
        ((kind == LocalScaleKind::AddCC || kind == LocalScaleKind::MulCC) &&
         !valid(b)))
      return std::nullopt;
    std::optional<LocalScalePlan> best;
    auto consider = [&](std::optional<LocalScalePlan> candidate) {
      if (candidate &&
          (!best || rank(*candidate, forecast) < rank(*best, forecast)))
        best = candidate;
    };
    if (kind == LocalScaleKind::MulCP) {
      consider(coefficient(a, waterline, editable));
      if (optimize && !std::getenv("AUDIT_CP_NATURAL")) {
        for (double target :
             {waterline + 1e-11, waterline + 10, waterline + 20}) {
          unsigned low = 0, high = anchor - a.level;
          auto plain = [&](unsigned r) {
            return target - a.scale + removed(a.level, a.level + r);
          };
          if (plain(high) < waterline)
            continue;
          while (low < high) {
            unsigned mid = low + (high - low) / 2;
            if (plain(mid) >= waterline)
              high = mid;
            else
              low = mid + 1;
          }
          consider(coefficient(a, plain(low), editable, low));
        }
      }
      return best;
    }
    if (kind == LocalScaleKind::AddCC) {
      int64_t base = std::max(a.level, b.level);
      double natural = std::max(a.scale, b.scale);
      auto baseline = join(a, b, natural, base, editable);
      if (!baseline) {
        int64_t level =
            std::max(minimumLevel(a, natural), minimumLevel(b, natural));
        baseline = join(a, b, natural, level, editable);
      }
      consider(baseline);
      if (optimize) {
        for (int64_t level : {base, base + 1}) {
          if (level > anchor)
            continue;
          double target = std::max({waterline + 1e-11, minimumScale(a, level),
                                    minimumScale(b, level)});
          consider(join(a, b, target, level, editable));
        }
        double target = waterline + 1e-11;
        int64_t level =
            std::max(minimumLevel(a, target), minimumLevel(b, target));
        consider(join(a, b, target, level, editable));
      }
      return best;
    }
    LocalScalePlan result;
    result.result = a;
    if (kind == LocalScaleKind::MulCC) {
      auto level = std::max(a.level, b.level);
      result.inputs[0].modswitch = level - a.level;
      result.inputs[1].modswitch = level - b.level;
      result.work = (level != a.level) + (level != b.level);
      result.result = {level, a.scale + b.scale, a.nominal + b.nominal};
      if (editable) {
        int side =
            a.flexible() && (!b.flexible() || a.edits <= b.edits) ? 0 : 1;
        const auto &operand = side ? b : a;
        if (operand.flexible() && operand.edits + 1 <= maxEdits) {
          result.result.minShift = operand.minShift;
          result.result.maxShift =
              std::min(operand.maxShift, margin(result.result));
          result.result.edits =
              operand.edits + 1 + (result.inputs[side].modswitch != 0);
          if (result.result.edits > maxEdits)
            clearFlex(result.result);
          result.flexOperand = side;
        }
      }
      normalize(result);
    } else {
      if (editable && a.flexible() && a.edits + 1 <= maxEdits)
        ++result.result.edits;
      else
        clearFlex(result.result);
      result.flexOperand = 0;
    }
    if (valid(result.result))
      consider(result);
    return best;
  }

private:
  const ModulusChain &chain;
  int64_t anchor;
  double waterline;
  int64_t budget;
  unsigned bootLevel;
  double bootScale;

  static void clearFlex(LocalScaleValue &value) {
    value.edits = 0;
    value.minShift = 1;
    value.maxShift = 0;
  }
  double capacity(int64_t level) const {
    return chain.capacity(anchor - level);
  }
  double margin(LocalScaleValue value) const {
    return capacity(value.level) - value.scale - 1;
  }
  bool valid(LocalScaleValue value) const {
    return value.level >= 0 && value.level <= anchor &&
           std::isfinite(value.scale) && value.scale >= 0 &&
           value.scale <= 1023 && value.scale < capacity(value.level);
  }
  double removed(int64_t begin, int64_t end) const {
    return chain.removedLog2(anchor - begin, end - begin);
  }
  void normalize(LocalScalePlan &plan) const {
    if (plan.result.level > anchor)
      return;
    auto normalized = chain.normalize(plan.result.scale,
                                      anchor - plan.result.level, waterline);
    plan.result.level += normalized.drops;
    plan.result.scale = normalized.scale;
    plan.resultRescales = normalized.drops;
    plan.work += 4 * normalized.drops;
    if (normalized.drops)
      plan.result.nominal = std::ceil(normalized.scale);
    if (plan.result.flexible()) {
      plan.result.edits += normalized.drops;
      if (plan.result.edits > maxEdits)
        clearFlex(plan.result);
    }
  }
  std::optional<LocalScalePlan>
  coefficient(LocalScaleValue input, double plain, bool editable,
              std::optional<unsigned> drops = {}) const {
    if (plain < waterline || plain > 1023)
      return std::nullopt;
    LocalScalePlan plan;
    plan.encodingScale = plain;
    plan.result = {input.level, input.scale + plain,
                   input.nominal + static_cast<int64_t>(std::ceil(plain))};
    if (!valid(plan.result))
      return std::nullopt;
    if (editable) {
      plan.result.minShift = waterline - plain;
      plan.result.maxShift = margin(plan.result);
      plan.result.edits = 1;
    }
    if (drops) {
      for (unsigned i = 0; i < *drops; ++i) {
        plan.result.scale =
            chain.rescale(plan.result.scale, anchor - plan.result.level);
        ++plan.result.level;
      }
      plan.resultRescales = *drops;
      plan.work = 4 * *drops;
      if (*drops)
        plan.result.nominal = std::ceil(plan.result.scale);
      plan.result.edits += editable ? *drops : 0;
      if (plan.result.edits > maxEdits)
        clearFlex(plan.result);
    } else
      normalize(plan);
    if (!valid(plan.result) || plan.result.scale < waterline)
      return std::nullopt;
    return plan;
  }

  std::optional<std::pair<LocalScaleAction, LocalScaleValue>>
  align(LocalScaleValue value, double target, int64_t level) const {
    if (level < value.level || level > anchor || target >= capacity(level))
      return std::nullopt;
    unsigned available = level - value.level;
    std::optional<std::pair<LocalScaleAction, LocalScaleValue>> best;
    auto attempt = [&](bool retune) {
      double minimum = retune ? value.minShift : auditUnitFloor(waterline);
      double maximum = retune ? value.maxShift : 1023;
      unsigned low = 0, high = available;
      auto shift = [&](unsigned r) {
        return target - value.scale + removed(level - r, level);
      };
      if (shift(high) < minimum)
        return;
      while (low < high) {
        unsigned mid = low + (high - low) / 2;
        if (shift(mid) >= minimum)
          high = mid;
        else
          low = mid + 1;
      }
      double delta = shift(low);
      if (delta > maximum)
        return;
      LocalScaleAction action{delta, available - low, low, retune, !retune};
      LocalScaleValue out = value;
      out.scale += delta;
      out.level = level - low;
      if (!valid(out))
        return;
      if (retune) {
        out.minShift -= delta;
        out.maxShift -= delta;
      } else {
        out.minShift = auditUnitFloor(waterline) - delta;
        out.maxShift = margin(out);
        out.edits = 1;
      }
      out.edits += low + (retune && action.modswitch != 0);
      if (out.edits > maxEdits)
        clearFlex(out);
      for (unsigned r = 0; r < low; ++r) {
        out.scale = chain.rescale(out.scale, anchor - out.level);
        ++out.level;
      }
      out.nominal = std::ceil(out.scale);
      if (!valid(out) || out.scale < waterline)
        return;
      auto work = [](const LocalScaleAction &a) {
        return 4 * a.rescales + 3 * a.unit + (a.modswitch != 0);
      };
      if (!best || work(action) < work(best->first))
        best = std::make_pair(action, out);
    };
    if (std::abs(target - value.scale) <= equalScale) {
      LocalScaleAction action;
      action.modswitch = available;
      auto out = value;
      out.level = level;
      if (out.flexible() && available) {
        ++out.edits;
        out.maxShift = std::min(out.maxShift, margin(out));
        if (out.edits > maxEdits)
          clearFlex(out);
      }
      if (valid(out))
        best = std::make_pair(action, out);
    }
    if (value.flexible())
      attempt(true);
    attempt(false);
    return best;
  }
  std::optional<LocalScalePlan> join(LocalScaleValue a, LocalScaleValue b,
                                     double target, int64_t level,
                                     bool editable) const {
    auto left = align(a, target, level), right = align(b, target, level);
    if (!left || !right)
      return std::nullopt;
    LocalScalePlan result;
    result.inputs = {left->first, right->first};
    a = left->second;
    b = right->second;
    result.result = {level, a.scale, std::max(a.nominal, b.nominal)};
    for (const auto &action : result.inputs)
      result.work +=
          4 * action.rescales + 3 * action.unit + (action.modswitch != 0);
    if (editable && a.flexible() && b.flexible() &&
        a.edits + b.edits + 1 <= maxEdits) {
      result.result.minShift = std::max(a.minShift, b.minShift);
      result.result.maxShift =
          std::min({a.maxShift, b.maxShift, margin(result.result) - 1});
      result.result.edits = a.edits + b.edits + 1;
    }
    return result;
  }
  double minimumScale(LocalScaleValue value, int64_t level) const {
    double bits = removed(value.level, level);
    double minimum = std::min(value.scale, value.scale + waterline - bits);
    if (value.flexible())
      minimum = std::min(minimum, value.scale + value.minShift - bits);
    return std::max(waterline + 1e-11, minimum);
  }
  int64_t minimumLevel(LocalScaleValue value, double target) const {
    if (std::abs(target - value.scale) <= equalScale)
      return value.level;
    int64_t low = value.level, high = anchor;
    if (target - value.scale + removed(value.level, high) < waterline)
      return anchor + 1;
    while (low < high) {
      int64_t mid = low + (high - low) / 2;
      if (target - value.scale + removed(value.level, mid) >= waterline)
        high = mid;
      else
        low = mid + 1;
    }
    return low;
  }
  bool refreshable(LocalScaleValue value) const {
    if (!valid(value) || value.level > budget)
      return false;
    return !bootScale ||
           planBootstrapInputAlignment(chain, anchor - value.level, value.scale,
                                       bootLevel, bootScale)
               .has_value();
  }
  std::tuple<bool, int64_t, unsigned, double, double>
  rank(const LocalScalePlan &plan,
       const std::array<int64_t, 3> &forecast) const {
    auto value = plan.result;
    unsigned work = plan.work;
    for (auto kind : forecast) {
      if (kind < 0 || value.level > anchor)
        break;
      LocalScalePlan next;
      next.result = value;
      next.result.scale = kind == 0 ? value.scale + waterline : value.scale * 2;
      normalize(next);
      value = next.result;
      work += next.work;
    }
    return std::make_tuple(!refreshable(plan.result), value.level, work,
                           value.scale, plan.result.scale);
  }
};
} // namespace hecate
#endif
