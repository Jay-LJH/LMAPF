#include "common.h"


std::ostream& operator<<(std::ostream& os, const Constraint& constraint)
{
	os << "<" << std::get<0>(constraint) << "," << std::get<1>(constraint) << "," <<
		std::get<2>(constraint) << "," << std::get<3>(constraint) << "," <<
		std::get<4>(constraint) << ">";
	return os;
}

std::ostream& operator<<(std::ostream& os, const Conflict& conflict)
{
	os << "<" << std::get<0>(conflict) << "," << std::get<1>(conflict) << "," <<
		std::get<2>(conflict) << "," << std::get<3>(conflict) << "," <<
		std::get<4>(conflict) << ">";
	return os;
}

ostream& operator<<(ostream& os, const Interval& interval)
{
    os << "[" << std::get<0>(interval) << "," << std::get<1>(interval) << ")(" <<
       std::get<2>(interval) << ")";
    return os;
}