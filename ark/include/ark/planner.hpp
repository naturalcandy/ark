// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_PLANNER_HPP
#define ARK_PLANNER_HPP

#include <functional>
#include <memory>
#include <string>

namespace ark {

class Model;

class DefaultPlanner {
   public:
    DefaultPlanner(const Model &model, int gpu_id);

    ~DefaultPlanner();

    using ConfigRule = std::function<std::string(const std::string &op,
                                                 const std::string &arch)>;

    void install_config_rule(ConfigRule rule);

    std::string plan(bool pretty = true) const;

   private:
    class Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace ark

#endif  // ARK_PLANNER_HPP
