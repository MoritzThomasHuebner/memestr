from collections import namedtuple


Event = namedtuple("Event", ["time_tag", "name", "label", "detectors", "duration"])

events = [
    Event(time_tag="1126259462-391", name="GW150914", label="GW150914", detectors="H1L1", duration=4),
    Event(time_tag="1128678900-4", name="GW151012", label="GW151012", detectors="H1L1", duration=4),
    Event(time_tag="1135136350-6", name="GW151226", label="GW151226", detectors="H1L1", duration=8),
    Event(time_tag="1167559936-6", name="GW170104", label="GW170104", detectors="H1L1", duration=4),
    Event(time_tag="1180922494-5", name="GW170608", label="GW170608", detectors="H1L1", duration=16),
    Event(time_tag="1185389807-3", name="GW170729", label="GW170729", detectors="H1L1V1", duration=4),
    Event(time_tag="1186302519-7", name="GW170809", label="GW170809", detectors="H1L1V1", duration=4),
    Event(time_tag="1186741861-5", name="GW170814", label="GW170814", detectors="H1L1V1", duration=4),
    Event(time_tag="1187058327-1", name="GW170818", label="GW170818", detectors="H1L1V1", duration=4),
    Event(time_tag="1187529256-5", name="GW170823", label="GW170823", detectors="H1L1", duration=4),
    Event(time_tag="1238782700-3", name="GW190408A", label="GW190408\_181802", detectors="H1L1V1", duration=4),
    Event(time_tag="1239082262-2", name="GW190412", label="GW190412", detectors="H1L1V1", duration=8),
    Event(time_tag="1239168612-5", name="GW190413A", label="GW190413\_052954", detectors="H1L1V1", duration=4),
    Event(time_tag="1239198206-7", name="GW190413B", label="GW190413\_134308", detectors="H1L1V1", duration=4),
    Event(time_tag="1239917954-3", name="GW190421A", label="GW190421\_213856", detectors="H1L1", duration=4),
    Event(time_tag="1240164426-1", name="GW190424A", label="GW190424\_180648", detectors="L1", duration=4),
    # Event(time_tag="1240327333-3", name="GW190426A", label="GW190426\_152155", detectors="H1L1V1", duration=64),
    Event(time_tag="1240944862-3", name="GW190503A", label="GW190503\_185404", detectors="H1L1V1", duration=4),
    Event(time_tag="1241719652-4", name="GW190512A", label="GW190512\_180714", detectors="H1L1V1", duration=4),
    Event(time_tag="1241816086-8", name="GW190513A", label="GW190513\_205428", detectors="H1L1V1", duration=4),
    Event(time_tag="1241852074-8", name="GW190514A", label="GW190514\_065416", detectors="H1L1", duration=4),
    Event(time_tag="1242107479-8", name="GW190517A", label="GW190517\_055101", detectors="H1L1V1", duration=4),
    Event(time_tag="1242315362-4", name="GW190519A", label="GW190519\_153544", detectors="H1L1V1", duration=4),
    Event(time_tag="1242442967-4", name="GW190521", label="GW190521", detectors="H1L1V1", duration=4),
    Event(time_tag="1242459857-5", name="GW190521A", label="GW190521\_074359", detectors="H1L1", duration=4),
    Event(time_tag="1242984073-8", name="GW190527A", label="GW190527\_092055", detectors="H1L1", duration=4),
    Event(time_tag="1243533585-1", name="GW190602A", label="GW190602\_175927", detectors="H1L1V1", duration=8),
    Event(time_tag="1245035079-3", name="GW190620A", label="GW190620\_030421", detectors="L1V1", duration=4),
    Event(time_tag="1245955943-2", name="GW190630A", label="GW190630\_185205", detectors="L1V1", duration=4),
    Event(time_tag="1246048404-6", name="GW190701A", label="GW190701\_203306", detectors="H1L1V1", duration=4),
    Event(time_tag="1246487219-3", name="GW190706A", label="GW190706\_222641", detectors="H1L1V1", duration=8),
    Event(time_tag="1246527224-2", name="GW190707A", label="GW190707\_093326", detectors="H1L1", duration=8),
    Event(time_tag="1246663515-4", name="GW190708A", label="GW190708\_232457", detectors="L1V1", duration=8),
    # Event(time_tag="1247608532-9", name="GW190719A", label="GW190719\_215514", detectors="H1L1", duration=4),
    Event(time_tag="1247616534-7", name="GW190720A", label="GW190720\_000836", detectors="H1L1V1", duration=8),
    Event(time_tag="1248242632-0", name="GW190727A", label="GW190727\_060333", detectors="H1L1V1", duration=4),
    Event(time_tag="1248331528-5", name="GW190728A", label="GW190728\_064510", detectors="H1L1V1", duration=8),
    Event(time_tag="1248617394-6", name="GW190731A", label="GW190731\_140936", detectors="H1L1", duration=4),
    Event(time_tag="1248834439-9", name="GW190803A", label="GW190803\_022701", detectors="H1L1V1", duration=4),
    Event(time_tag="1249852257-0", name="GW190814", label="GW190814", detectors="L1V1", duration=16),
    Event(time_tag="1251009263-8", name="GW190828A", label="GW190828\_063405", detectors="H1L1V1", duration=4),
    Event(time_tag="1251010527-9", name="GW190828B", label="GW190828\_065509", detectors="H1L1V1", duration=8),
    # Event(time_tag="1252064527-7", name="GW190909A", label="GW190909\_114149", detectors="H1L1", duration=4),
    Event(time_tag="1252150105-3", name="GW190910A", label="GW190910\_112807", detectors="L1V1", duration=4),
    Event(time_tag="1252627040-7", name="GW190915A", label="GW190915\_235702", detectors="H1L1V1", duration=4),
    Event(time_tag="1253326744-8", name="GW190924A", label="GW190924\_021846", detectors="H1L1V1", duration=16),
    Event(time_tag="1253755327-5", name="GW190929A", label="GW190929\_012149", detectors="H1L1V1", duration=4),
    Event(time_tag="1253885759-2", name="GW190930A", label="GW190930\_133541", detectors="H1L1", duration=8),
]

precessing_events = [
    Event(time_tag="1126259462-391", name="GW150914_prec", label="GW150914", detectors="H1L1", duration=4),
    Event(time_tag="1128678900-4", name="GW151012_prec", label="GW151012", detectors="H1L1", duration=4),
    Event(time_tag="1135136350-6", name="GW151226_prec", label="GW151226", detectors="H1L1", duration=8),
    Event(time_tag="1167559936-6", name="GW170104_prec", label="GW170104", detectors="H1L1", duration=4),
    Event(time_tag="1180922494-5", name="GW170608_prec", label="GW170608", detectors="H1L1", duration=16),
    Event(time_tag="1185389807-3", name="GW170729_prec", label="GW170729", detectors="H1L1V1", duration=4),
    Event(time_tag="1186302519-7", name="GW170809_prec", label="GW170809", detectors="H1L1V1", duration=4),
    Event(time_tag="1186741861-5", name="GW170814_prec", label="GW170814", detectors="H1L1V1", duration=4),
    Event(time_tag="1187058327-1", name="GW170818_prec", label="GW170818", detectors="H1L1V1", duration=4),
    Event(time_tag="1187529256-5", name="GW170823_prec", label="GW170823", detectors="H1L1", duration=4),
    Event(time_tag="1238782700-3", name="GW190408A_prec", label="GW190408\_181802", detectors="H1L1V1", duration=4),
    Event(time_tag="1239168612-5", name="GW190413A_prec", label="GW190413\_052954", detectors="H1L1V1", duration=4),
    Event(time_tag="1239198206-7", name="GW190413B_prec", label="GW190413\_134308", detectors="H1L1V1", duration=4),
    Event(time_tag="1239917954-3", name="GW190421A_prec", label="GW190421\_213856", detectors="H1L1", duration=4),
    Event(time_tag="1240164426-1", name="GW190424A_prec", label="GW190424\_180648", detectors="L1", duration=4),
    Event(time_tag="1240944862-3", name="GW190503A_prec", label="GW190503\_185404", detectors="H1L1V1", duration=4),
    Event(time_tag="1241719652-4", name="GW190512A_prec", label="GW190512\_180714", detectors="H1L1V1", duration=4),
    Event(time_tag="1241816086-8", name="GW190513A_prec", label="GW190513\_205428", detectors="H1L1V1", duration=4),
    Event(time_tag="1241852074-8", name="GW190514A_prec", label="GW190514\_065416", detectors="H1L1", duration=4),
    Event(time_tag="1242107479-8", name="GW190517A_prec", label="GW190517\_055101", detectors="H1L1V1", duration=4),
    Event(time_tag="1242315362-4", name="GW190519A_prec", label="GW190519\_153544", detectors="H1L1V1", duration=4),
    Event(time_tag="1242442967-4", name="GW190521_prec", label="GW190521", detectors="H1L1V1", duration=8),
    Event(time_tag="1242459857-5", name="GW190521A_prec", label="GW190521\_074359", detectors="H1L1", duration=4),
    Event(time_tag="1242984073-8", name="GW190527A_prec", label="GW190527\_092055", detectors="H1L1", duration=4),
    Event(time_tag="1243533585-1", name="GW190602A_prec", label="GW190602\_175927", detectors="H1L1V1", duration=8),
    Event(time_tag="1245035079-3", name="GW190620A_prec", label="GW190620\_030421", detectors="L1V1", duration=4),
    Event(time_tag="1245955943-2", name="GW190630A_prec", label="GW190630\_185205", detectors="L1V1", duration=4),
    Event(time_tag="1246048404-6", name="GW190701A_prec", label="GW190701\_203306", detectors="H1L1V1", duration=4),
    Event(time_tag="1246487219-3", name="GW190706A_prec", label="GW190706\_222641", detectors="H1L1V1", duration=8),
    Event(time_tag="1246527224-2", name="GW190707A_prec", label="GW190707\_093326", detectors="H1L1", duration=8),
    Event(time_tag="1246663515-4", name="GW190708A_prec", label="GW190708\_232457", detectors="L1V1", duration=8),
    # Event(time_tag="1247608532-9", name="GW190719A_prec", label="GW190719\_215514", detectors="H1L1", duration=4),
    Event(time_tag="1247616534-7", name="GW190720A_prec", label="GW190720\_000836", detectors="H1L1V1", duration=8),
    Event(time_tag="1248242632-0", name="GW190727A_prec", label="GW190727\_060333", detectors="H1L1V1", duration=4),
    Event(time_tag="1248331528-5", name="GW190728A_prec", label="GW190728\_064510", detectors="H1L1V1", duration=8),
    Event(time_tag="1248617394-6", name="GW190731A_prec", label="GW190731\_140936", detectors="H1L1", duration=4),
    Event(time_tag="1248834439-9", name="GW190803A_prec", label="GW190803\_022701", detectors="H1L1V1", duration=4),
    Event(time_tag="1251009263-8", name="GW190828A_prec", label="GW190828\_063405", detectors="H1L1V1", duration=4),
    Event(time_tag="1251010527-9", name="GW190828B_prec", label="GW190828\_065509", detectors="H1L1V1", duration=8),
    # Event(time_tag="1252064527-7", name="GW190909A_prec", label="GW190909\_114149", detectors="H1L1", duration=4),
    Event(time_tag="1252150105-3", name="GW190910A_prec", label="GW190910\_112807", detectors="L1V1", duration=4),
    Event(time_tag="1252627040-7", name="GW190915A_prec", label="GW190915\_235702", detectors="H1L1V1", duration=4),
    Event(time_tag="1253326744-8", name="GW190924A_prec", label="GW190924\_021846", detectors="H1L1V1", duration=16),
    Event(time_tag="1253755327-5", name="GW190929A_prec", label="GW190929\_012149", detectors="H1L1V1", duration=4),
    Event(time_tag="1253885759-2", name="GW190930A_prec", label="GW190930\_133541", detectors="H1L1", duration=8)

]
