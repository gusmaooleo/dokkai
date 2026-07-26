"""
Synthetic eval fixture (roadmap feature 18, part 1, commit C2) — a small,
100% FICTIONAL "greenhouse automation" codebase used ONLY to exercise
``scripts/eval_retrieval.py`` deterministically, without touching any real
(private) corpus. Nothing here resembles any customer codebase.

``build_chunks()`` returns plain ``CodeChunk`` objects (the same dataclass
``services.chunker.chunk_graph`` produces from a real code-graph-rag run) —
this module bypasses graph extraction entirely and constructs the chunks by
hand, so ``scripts/eval_retrieval.py`` can self-seed it into Weaviate via the
same ``upsert_chunks`` plumbing real ingestion uses.

Deliberately engineered to exercise:
  - identifier shapes: snake_case, camelCase, dotted/qualified name, a bare
    Class name;
  - described AND undescribed entities (``description`` set directly on some
    chunks — no LLM describe pass; an empty ``description`` AND ``doc``
    means no ``summary`` vector gets created, per ``weaviate_client``);
  - a real 3-hop call chain (handle_watering_request -> get_zone_status ->
    read_soil_moisture -> read_raw_value) for flow questions;
  - test files (``src/tests/...``), each repeating one keyword many times,
    to exercise the BM25 test-file penalty;
  - a hub node (``NotificationService.notify``, 20 callers — capped at 16 in
    the rendered ``chunk_text``, full list still in the stored property);
  - a contained Class + its own Method in the same file (containment dedup
    in ``Retriever.build_graph_context``): ``IrrigationController`` /
    ``IrrigationController.handle_watering_request``;
  - entities with zero lexical overlap with the negation probes in
    ``scripts/eval_dataset_fixture.json`` (chess, mortgages, Shakespeare,
    baking — nothing greenhouse-related).

All source snippets are static strings — no randomness, no timestamps, so
the corpus (and the resulting embeddings/rankings) is fully deterministic
across runs.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from services.chunker import CodeChunk  # noqa: E402

PROJECT_NAME = "_eval_fixture"

_node_id = itertools.count(1)


def _chunk(
    entity_type: str,
    name: str,
    qualified_name: str,
    file_path: str,
    module_name: str,
    *,
    parent_class: str | None = None,
    start_line: int = 1,
    end_line: int = 10,
    calls: list[str] | None = None,
    called_by: list[str] | None = None,
    defined_methods: list[str] | None = None,
    description: str = "",
    source_code: str = "",
) -> CodeChunk:
    chunk = CodeChunk(
        node_id=next(_node_id),
        entity_type=entity_type,
        name=name,
        qualified_name=qualified_name,
        imports=[],
        file_path=file_path,
        absolute_path=f"/repo/{file_path}",
        start_line=start_line,
        end_line=end_line,
        project_name=PROJECT_NAME,
        module_name=module_name,
        parent_class=parent_class,
        calls=calls or [],
        called_by=called_by or [],
        defined_methods=defined_methods or [],
        source_code=source_code,
        description=description,
    )
    chunk.build_text()  # populate chunk_text, same as chunk_graph() does
    return chunk


def build_chunks() -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    # -- controllers/irrigation_controller.py --------------------------------
    # Class + its own Method in the SAME file, method range nested inside the
    # class range -> exercises _drop_contained (containment dedup).
    chunks.append(_chunk(
        "Class", "IrrigationController",
        "greenhouse.controllers.irrigation_controller.IrrigationController",
        "src/controllers/irrigation_controller.py",
        "greenhouse.controllers.irrigation_controller",
        start_line=1, end_line=80,
        defined_methods=["handle_watering_request", "schedule_watering_cycle"],
        description="Coordinates watering requests for a single greenhouse zone.",
        source_code=(
            "class IrrigationController:\n"
            "    def __init__(self, zone_service, pump_controller, notifier):\n"
            "        self.zone_service = zone_service\n"
            "        self.pump_controller = pump_controller\n"
            "        self.notifier = notifier\n"
        ),
    ))
    chunks.append(_chunk(
        "Method", "handle_watering_request",
        "greenhouse.controllers.irrigation_controller.IrrigationController.handle_watering_request",
        "src/controllers/irrigation_controller.py",
        "greenhouse.controllers.irrigation_controller",
        parent_class="greenhouse.controllers.irrigation_controller.IrrigationController",
        start_line=20, end_line=45,
        calls=[
            "greenhouse.services.zone_service.ZoneService.get_zone_status",
            "greenhouse.controllers.pump_controller.activate_water_pump",
            "greenhouse.services.notification_service.NotificationService.notify",
        ],
        description="Handles an incoming watering request: checks the zone's soil "
                     "moisture status, activates the water pump if needed, and "
                     "notifies the operator once the cycle is scheduled.",
        source_code=(
            "def handle_watering_request(self, zone_id):\n"
            "    status = self.zone_service.get_zone_status(zone_id)\n"
            "    if status.needs_water:\n"
            "        activate_water_pump(zone_id)\n"
            "        self.notifier.notify(f'watering zone {zone_id}')\n"
            "    return status\n"
        ),
    ))
    chunks.append(_chunk(
        "Method", "schedule_watering_cycle",
        "greenhouse.controllers.irrigation_controller.IrrigationController.schedule_watering_cycle",
        "src/controllers/irrigation_controller.py",
        "greenhouse.controllers.irrigation_controller",
        parent_class="greenhouse.controllers.irrigation_controller.IrrigationController",
        start_line=47, end_line=60,
        calls=["greenhouse.controllers.irrigation_controller.IrrigationController.handle_watering_request"],
        source_code=(
            "def schedule_watering_cycle(self, zone_ids):\n"
            "    for zone_id in zone_ids:\n"
            "        self.handle_watering_request(zone_id)\n"
        ),
    ))

    # -- controllers/pump_controller.py --------------------------------------
    chunks.append(_chunk(
        "Function", "activate_water_pump",
        "greenhouse.controllers.pump_controller.activate_water_pump",
        "src/controllers/pump_controller.py",
        "greenhouse.controllers.pump_controller",
        start_line=1, end_line=6,
        called_by=["greenhouse.controllers.irrigation_controller.IrrigationController.handle_watering_request"],
        description="Opens the solenoid valve and switches the zone's water pump on.",
        source_code=(
            "def activate_water_pump(zone_id):\n"
            "    valve = get_valve(zone_id)\n"
            "    valve.open()\n"
            "    return 'water_pump_on'\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "deactivate_water_pump",
        "greenhouse.controllers.pump_controller.deactivate_water_pump",
        "src/controllers/pump_controller.py",
        "greenhouse.controllers.pump_controller",
        start_line=8, end_line=13,
        source_code=(
            "def deactivate_water_pump(zone_id):\n"
            "    valve = get_valve(zone_id)\n"
            "    valve.close()\n"
            "    return 'water_pump_off'\n"
        ),
    ))

    # -- controllers/climate_controller.py -----------------------------------
    chunks.append(_chunk(
        "Class", "ClimateController",
        "greenhouse.controllers.climate_controller.ClimateController",
        "src/controllers/climate_controller.py",
        "greenhouse.controllers.climate_controller",
        start_line=1, end_line=30,
        defined_methods=["adjust_temperature"],
        description="Keeps a greenhouse bay within its target temperature range.",
        source_code="class ClimateController:\n    def __init__(self, actuators):\n        self.actuators = actuators\n",
    ))
    chunks.append(_chunk(
        "Method", "adjust_temperature",
        "greenhouse.controllers.climate_controller.ClimateController.adjust_temperature",
        "src/controllers/climate_controller.py",
        "greenhouse.controllers.climate_controller",
        parent_class="greenhouse.controllers.climate_controller.ClimateController",
        start_line=5, end_line=20,
        calls=["greenhouse.services.actuator_service.set_vent_position"],
        description="Compares the current bay temperature against its target and "
                     "opens or closes the roof vents to correct it.",
        source_code=(
            "def adjust_temperature(self, bay_id, current, target):\n"
            "    if current > target:\n"
            "        set_vent_position(bay_id, 'open')\n"
            "    else:\n"
            "        set_vent_position(bay_id, 'closed')\n"
        ),
    ))

    # -- services/zone_service.py ---------------------------------------------
    chunks.append(_chunk(
        "Class", "ZoneService",
        "greenhouse.services.zone_service.ZoneService",
        "src/services/zone_service.py",
        "greenhouse.services.zone_service",
        start_line=1, end_line=40,
        defined_methods=["get_zone_status", "list_active_zones"],
        description="Tracks the current status (moisture, temperature) of every "
                     "irrigation zone in a greenhouse.",
        source_code="class ZoneService:\n    def __init__(self, sensor_service):\n        self.sensor_service = sensor_service\n",
    ))
    chunks.append(_chunk(
        "Method", "get_zone_status",
        "greenhouse.services.zone_service.ZoneService.get_zone_status",
        "src/services/zone_service.py",
        "greenhouse.services.zone_service",
        parent_class="greenhouse.services.zone_service.ZoneService",
        start_line=10, end_line=20,
        calls=["greenhouse.services.sensor_service.read_soil_moisture"],
        called_by=["greenhouse.controllers.irrigation_controller.IrrigationController.handle_watering_request"],
        description="Reads the soil moisture sensor for a zone and returns whether "
                     "the zone currently needs water.",
        source_code=(
            "def get_zone_status(self, zone_id):\n"
            "    moisture = read_soil_moisture(zone_id)\n"
            "    return ZoneStatus(needs_water=moisture < 30)\n"
        ),
    ))
    chunks.append(_chunk(
        "Method", "list_active_zones",
        "greenhouse.services.zone_service.ZoneService.list_active_zones",
        "src/services/zone_service.py",
        "greenhouse.services.zone_service",
        parent_class="greenhouse.services.zone_service.ZoneService",
        start_line=22, end_line=28,
        source_code="def list_active_zones(self):\n    return [z for z in self.zones if z.active]\n",
    ))

    # -- services/sensor_service.py --------------------------------------------
    chunks.append(_chunk(
        "Function", "read_soil_moisture",
        "greenhouse.services.sensor_service.read_soil_moisture",
        "src/services/sensor_service.py",
        "greenhouse.services.sensor_service",
        start_line=1, end_line=8,
        calls=["greenhouse.drivers.moisture_sensor_driver.read_raw_value"],
        called_by=["greenhouse.services.zone_service.ZoneService.get_zone_status"],
        description="Reads the soil moisture level for a zone from its moisture "
                     "sensor driver and converts the raw reading to a percentage.",
        source_code=(
            "def read_soil_moisture(zone_id):\n"
            "    raw = read_raw_value(zone_id)\n"
            "    return raw_to_percentage(raw)\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "read_ambient_temperature",
        "greenhouse.services.sensor_service.read_ambient_temperature",
        "src/services/sensor_service.py",
        "greenhouse.services.sensor_service",
        start_line=10, end_line=15,
        source_code="def read_ambient_temperature(bay_id):\n    return temperature_sensor(bay_id).read()\n",
    ))

    # -- drivers/moisture_sensor_driver.py --------------------------------------
    chunks.append(_chunk(
        "Function", "read_raw_value",
        "greenhouse.drivers.moisture_sensor_driver.read_raw_value",
        "src/drivers/moisture_sensor_driver.py",
        "greenhouse.drivers.moisture_sensor_driver",
        start_line=1, end_line=6,
        called_by=["greenhouse.services.sensor_service.read_soil_moisture"],
        description="Reads the raw analog-to-digital value from the physical "
                     "capacitive moisture sensor over the I2C bus.",
        source_code="def read_raw_value(zone_id):\n    return i2c_bus.read(zone_id, register=0x00)\n",
    ))
    chunks.append(_chunk(
        "Function", "calibrate_sensor",
        "greenhouse.drivers.moisture_sensor_driver.calibrate_sensor",
        "src/drivers/moisture_sensor_driver.py",
        "greenhouse.drivers.moisture_sensor_driver",
        start_line=8, end_line=13,
        source_code="def calibrate_sensor(zone_id, dry_value, wet_value):\n    store_calibration(zone_id, dry_value, wet_value)\n",
    ))

    # -- services/nutrient_service.py (camelCase identifiers) -------------------
    chunks.append(_chunk(
        "Function", "calculateNutrientMix",
        "greenhouse.services.nutrient_service.calculateNutrientMix",
        "src/services/nutrient_service.py",
        "greenhouse.services.nutrient_service",
        start_line=1, end_line=10,
        calls=["greenhouse.services.nutrient_service.mixFertilizerBatch"],
        description="Calculates the nitrogen/phosphorus/potassium ratio needed for "
                     "a crop cycle and hands it off to be mixed into a batch.",
        source_code=(
            "def calculateNutrientMix(crop_type, volume_liters):\n"
            "    ratio = npk_ratio_for(crop_type)\n"
            "    return mixFertilizerBatch(ratio, volume_liters)\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "mixFertilizerBatch",
        "greenhouse.services.nutrient_service.mixFertilizerBatch",
        "src/services/nutrient_service.py",
        "greenhouse.services.nutrient_service",
        start_line=12, end_line=20,
        called_by=["greenhouse.services.nutrient_service.calculateNutrientMix"],
        description="Mixes a fertilizer batch to the given NPK ratio and volume, "
                     "then logs the batch for traceability.",
        source_code=(
            "def mixFertilizerBatch(ratio, volume_liters):\n"
            "    batch = FertilizerBatch(ratio, volume_liters)\n"
            "    logNutrientBatch(batch)\n"
            "    return batch\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "logNutrientBatch",
        "greenhouse.services.nutrient_service.logNutrientBatch",
        "src/services/nutrient_service.py",
        "greenhouse.services.nutrient_service",
        start_line=22, end_line=27,
        source_code="def logNutrientBatch(batch):\n    audit_log.append(batch)\n",
    ))

    # -- services/notification_service.py (hub node) -----------------------------
    chunks.append(_chunk(
        "Class", "NotificationService",
        "greenhouse.services.notification_service.NotificationService",
        "src/services/notification_service.py",
        "greenhouse.services.notification_service",
        start_line=1, end_line=40,
        defined_methods=["notify", "notify_admin"],
        description="Delivers operator-facing notifications (SMS, push, email) "
                     "about greenhouse events.",
        source_code="class NotificationService:\n    def __init__(self, channels):\n        self.channels = channels\n",
    ))
    # 20 fictional callers -- deterministic, generated (not hand-typed) -- to
    # exercise the hub-node capped-list rendering (_MAX_RELATION_ITEMS=16 in
    # chunk_text; the full 20-item list stays in the stored `called_by` array).
    _hub_callers = [f"greenhouse.integrations.legacy_caller_{i:02d}.invoke_notify" for i in range(1, 21)]
    chunks.append(_chunk(
        "Method", "notify",
        "greenhouse.services.notification_service.NotificationService.notify",
        "src/services/notification_service.py",
        "greenhouse.services.notification_service",
        parent_class="greenhouse.services.notification_service.NotificationService",
        start_line=5, end_line=15,
        called_by=_hub_callers,
        description="Sends a notification message to every configured channel "
                     "(SMS, push, email) for the current operator.",
        source_code="def notify(self, message):\n    for channel in self.channels:\n        channel.send(message)\n",
    ))
    chunks.append(_chunk(
        "Method", "notify_admin",
        "greenhouse.services.notification_service.NotificationService.notify_admin",
        "src/services/notification_service.py",
        "greenhouse.services.notification_service",
        parent_class="greenhouse.services.notification_service.NotificationService",
        start_line=17, end_line=22,
        source_code="def notify_admin(self, message):\n    self.channels['admin'].send(message)\n",
    ))

    # -- services/actuator_service.py --------------------------------------------
    chunks.append(_chunk(
        "Function", "set_vent_position",
        "greenhouse.services.actuator_service.set_vent_position",
        "src/services/actuator_service.py",
        "greenhouse.services.actuator_service",
        start_line=1, end_line=6,
        called_by=["greenhouse.controllers.climate_controller.ClimateController.adjust_temperature"],
        source_code="def set_vent_position(bay_id, position):\n    vent_actuator(bay_id).move_to(position)\n",
    ))
    chunks.append(_chunk(
        "Function", "set_heater_power",
        "greenhouse.services.actuator_service.set_heater_power",
        "src/services/actuator_service.py",
        "greenhouse.services.actuator_service",
        start_line=8, end_line=13,
        description="Sets the greenhouse bay's heater output as a percentage of "
                     "its rated power.",
        source_code="def set_heater_power(bay_id, percent):\n    heater_actuator(bay_id).set_power(percent)\n",
    ))

    # -- reporting/report_generator.py --------------------------------------------
    chunks.append(_chunk(
        "Class", "ReportGenerator",
        "greenhouse.reporting.report_generator.ReportGenerator",
        "src/reporting/report_generator.py",
        "greenhouse.reporting.report_generator",
        start_line=1, end_line=30,
        defined_methods=["generate_daily_summary", "format_report"],
        description="Builds end-of-day operational summaries across all zones "
                     "and bays of a greenhouse.",
        source_code="class ReportGenerator:\n    def __init__(self, zone_service):\n        self.zone_service = zone_service\n",
    ))
    chunks.append(_chunk(
        "Method", "generate_daily_summary",
        "greenhouse.reporting.report_generator.ReportGenerator.generate_daily_summary",
        "src/reporting/report_generator.py",
        "greenhouse.reporting.report_generator",
        parent_class="greenhouse.reporting.report_generator.ReportGenerator",
        start_line=5, end_line=18,
        calls=["greenhouse.reporting.report_generator.ReportGenerator.format_report"],
        description="Gathers today's watering and climate events for every zone "
                     "and formats them into a single daily summary report.",
        source_code=(
            "def generate_daily_summary(self):\n"
            "    events = self.zone_service.list_active_zones()\n"
            "    return self.format_report(events)\n"
        ),
    ))
    chunks.append(_chunk(
        "Method", "format_report",
        "greenhouse.reporting.report_generator.ReportGenerator.format_report",
        "src/reporting/report_generator.py",
        "greenhouse.reporting.report_generator",
        parent_class="greenhouse.reporting.report_generator.ReportGenerator",
        start_line=20, end_line=27,
        source_code="def format_report(self, events):\n    return '\\n'.join(str(e) for e in events)\n",
    ))

    # -- utils/config_loader.py ----------------------------------------------------
    chunks.append(_chunk(
        "Function", "load_config",
        "greenhouse.utils.config_loader.load_config",
        "src/utils/config_loader.py",
        "greenhouse.utils.config_loader",
        start_line=1, end_line=6,
        source_code="def load_config(path):\n    return yaml.safe_load(open(path))\n",
    ))
    chunks.append(_chunk(
        "Function", "validate_config",
        "greenhouse.utils.config_loader.validate_config",
        "src/utils/config_loader.py",
        "greenhouse.utils.config_loader",
        start_line=8, end_line=14,
        source_code="def validate_config(config):\n    return 'zones' in config and 'bays' in config\n",
    ))

    # -- utils/logger.py -------------------------------------------------------------
    chunks.append(_chunk(
        "Function", "log_event",
        "greenhouse.utils.logger.log_event",
        "src/utils/logger.py",
        "greenhouse.utils.logger",
        start_line=1, end_line=5,
        description="Appends a timestamped event line to the greenhouse operations log.",
        source_code="def log_event(message):\n    operations_log.append(message)\n",
    ))

    # -- scheduler/greenhouse_scheduler.py --------------------------------------------
    chunks.append(_chunk(
        "Class", "GreenhouseScheduler",
        "greenhouse.scheduler.greenhouse_scheduler.GreenhouseScheduler",
        "src/scheduler/greenhouse_scheduler.py",
        "greenhouse.scheduler.greenhouse_scheduler",
        start_line=1, end_line=30,
        defined_methods=["run_daily_tasks"],
        description="Runs the daily batch of greenhouse automation tasks: "
                     "watering cycles and the end-of-day report.",
        source_code="class GreenhouseScheduler:\n    def __init__(self, irrigation, reports):\n        self.irrigation = irrigation\n        self.reports = reports\n",
    ))
    chunks.append(_chunk(
        "Method", "run_daily_tasks",
        "greenhouse.scheduler.greenhouse_scheduler.GreenhouseScheduler.run_daily_tasks",
        "src/scheduler/greenhouse_scheduler.py",
        "greenhouse.scheduler.greenhouse_scheduler",
        parent_class="greenhouse.scheduler.greenhouse_scheduler.GreenhouseScheduler",
        start_line=5, end_line=20,
        calls=[
            "greenhouse.controllers.irrigation_controller.IrrigationController.schedule_watering_cycle",
            "greenhouse.reporting.report_generator.ReportGenerator.generate_daily_summary",
        ],
        description="Kicks off every zone's watering cycle for the day, then "
                     "generates and stores the daily summary report.",
        source_code=(
            "def run_daily_tasks(self, zone_ids):\n"
            "    self.irrigation.schedule_watering_cycle(zone_ids)\n"
            "    return self.reports.generate_daily_summary()\n"
        ),
    ))

    # -- services/alert_service.py ----------------------------------------------------
    chunks.append(_chunk(
        "Function", "raise_low_moisture_alert",
        "greenhouse.services.alert_service.raise_low_moisture_alert",
        "src/services/alert_service.py",
        "greenhouse.services.alert_service",
        start_line=1, end_line=8,
        calls=["greenhouse.services.notification_service.NotificationService.notify"],
        description="Raises a low-moisture alert for a zone and delivers it as a "
                     "notification to the operator.",
        source_code=(
            "def raise_low_moisture_alert(zone_id, notifier):\n"
            "    notifier.notify(f'zone {zone_id} moisture is critically low')\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "clear_alert",
        "greenhouse.services.alert_service.clear_alert",
        "src/services/alert_service.py",
        "greenhouse.services.alert_service",
        start_line=10, end_line=14,
        source_code="def clear_alert(alert_id):\n    active_alerts.pop(alert_id, None)\n",
    ))

    # -- models/soil_moisture_model.py -------------------------------------------------
    chunks.append(_chunk(
        "Class", "SoilMoistureModel",
        "greenhouse.models.soil_moisture_model.SoilMoistureModel",
        "src/models/soil_moisture_model.py",
        "greenhouse.models.soil_moisture_model",
        start_line=1, end_line=20,
        defined_methods=["to_dict"],
        description="A single soil-moisture reading for a zone: percentage, "
                     "timestamp, and the sensor that produced it.",
        source_code="class SoilMoistureModel:\n    def __init__(self, zone_id, percent, sensor_id):\n        self.zone_id = zone_id\n        self.percent = percent\n        self.sensor_id = sensor_id\n",
    ))
    chunks.append(_chunk(
        "Method", "to_dict",
        "greenhouse.models.soil_moisture_model.SoilMoistureModel.to_dict",
        "src/models/soil_moisture_model.py",
        "greenhouse.models.soil_moisture_model",
        parent_class="greenhouse.models.soil_moisture_model.SoilMoistureModel",
        start_line=10, end_line=16,
        source_code="def to_dict(self):\n    return {'zone_id': self.zone_id, 'percent': self.percent}\n",
    ))

    # -- test files: exercise the BM25 test-file penalty ---------------------------
    # Each test repeats its probe term several times in the body, so its raw
    # BM25 term-frequency score would out-rank the non-test implementation
    # without RETRIEVAL_TEST_PENALTY (see services/retriever.py::_is_test_path).
    chunks.append(_chunk(
        "Function", "test_activate_water_pump",
        "greenhouse.tests.test_pump_controller.test_activate_water_pump",
        "src/tests/test_pump_controller.py",
        "greenhouse.tests.test_pump_controller",
        start_line=1, end_line=8,
        calls=["greenhouse.controllers.pump_controller.activate_water_pump"],
        source_code=(
            "def test_activate_water_pump():\n"
            "    # water_pump water_pump water_pump water_pump\n"
            "    assert activate_water_pump('zone-1') == 'water_pump_on'\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "test_get_zone_status",
        "greenhouse.tests.test_zone_service.test_get_zone_status",
        "src/tests/test_zone_service.py",
        "greenhouse.tests.test_zone_service",
        start_line=1, end_line=8,
        calls=["greenhouse.services.zone_service.ZoneService.get_zone_status"],
        source_code=(
            "def test_get_zone_status():\n"
            "    # zone_status zone_status zone_status zone_status\n"
            "    assert get_zone_status('zone-1').needs_water in (True, False)\n"
        ),
    ))
    chunks.append(_chunk(
        "Function", "test_notify",
        "greenhouse.tests.test_notification_service.test_notify",
        "src/tests/test_notification_service.py",
        "greenhouse.tests.test_notification_service",
        start_line=1, end_line=8,
        calls=["greenhouse.services.notification_service.NotificationService.notify"],
        source_code=(
            "def test_notify():\n"
            "    # notify notify notify notify\n"
            "    assert notify('hello') is None\n"
        ),
    ))

    return chunks
