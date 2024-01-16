// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ARK_RANGE_H_
#define ARK_RANGE_H_

namespace ark {

template <typename T = int>
class Range {
   public:
    Range() : begin_(0), end_(0), step_(1) {}
    Range(T begin, T end, T step = 1) : begin_(begin), end_(end), step_(step) {}

    class Iterator {
       public:
        Iterator(T value, T begin, T end, T step)
            : value_(value), begin_(begin), end_(end), step_(step) {}

        Iterator &operator++() {
            value_ += step_;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator &other) const {
            if (begin_ < end_ && value_ >= end_) {
                return other.value_ >= end_;
            }
            if (begin_ > end_ && value_ <= end_) {
                return other.value_ <= end_;
            }
            return value_ == other.value_;
        }

        bool operator!=(const Iterator &other) const {
            return !(*this == other);
        }

        T operator*() const { return value_; }

       private:
        T value_;
        T begin_;
        T end_;
        T step_;
    };

    Iterator begin() const { return Iterator(begin_, begin_, end_, step_); }

    Iterator end() const { return Iterator(end_, begin_, end_, step_); }

    T step() const { return step_; }

   private:
    T begin_;
    T end_;
    T step_;
};

template <typename T = int>
Range<T> range(T end) {
    return Range<T>(0, end);
}

template <typename T = int>
Range<T> range(T begin, T end) {
    return Range<T>(begin, end);
}

template <typename T = int>
Range<T> range(T begin, T end, T step) {
    return Range<T>(begin, end, step);
}

}  // namespace ark

#endif  // ARK_RANGE_H_
