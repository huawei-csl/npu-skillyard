#pragma once

// CANN 9.0 AICORE boilerplate (bundled with the npu-skillyard plugin).
// Device pass  (bisheng -xcce):      PTO headers define AICORE = [aicore]
// Host pass    (structural validate): this header defines AICORE = [aicore]
// CPU-SIM pass (-D__CPU_SIM):         PTO cpu_stub provides the ACL stubs
// DO NOT edit AICORE in kernel files -- include this header and use AICORE.

// Host-side includes (real on host + device passes; stubbed under CPU-SIM).
#ifndef __CPU_SIM
#include "acl/acl.h"
#include <runtime/runtime/rt_ffts.h>
#endif
#include <cmath>
#include <cstdint>

// PTO ISA include + namespace (device pass and CPU-SIM pass).
#if defined(__CCE_AICORE__)
#include <pto/pto-inst.hpp>
using namespace pto;
#elif defined(__CPU_SIM)
#include <pto/pto-inst.hpp>
using namespace pto;
#endif

// AICORE fallback for the host pass (the PTO header handles the device pass).
#ifndef AICORE
#define AICORE [aicore]
#endif
