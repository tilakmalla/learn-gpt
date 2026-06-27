"""
STEP 1: Download Physics Training Dataset
==========================================
This script downloads/generates the physics dataset for training the GPT model.

The physics dataset contains explanations of fundamental physics concepts including:
- Classical Mechanics (Newton's laws, momentum, energy)
- Thermodynamics (heat, entropy, temperature)
- Waves and Oscillations (sound, light, interference)
- Electricity and Magnetism (charge, current, fields)
- Modern Physics (relativity, quantum mechanics)
- Astrophysics (stars, galaxies, cosmology)

USAGE:
    python download_dataset.py              # Generate physics dataset

AFTER DOWNLOADING:
    python data_pipeline.py    # Test tokenization (Phase 2)
    python gpt_model.py        # Test model architecture (Phase 3)
    python train.py            # Train the model (Phase 4) ~20-30 min
    python generate.py -i      # Generate text (Phase 5)

EXAMPLE PROMPTS:
    "Newton", "Energy", "The force", "Momentum", "Gravity", "Quantum"
"""

import os
import argparse


def generate_physics_dataset(filepath: str):
    """Generate a physics concepts dataset."""
    
    # Physics text covering various topics
    physics_text = '''
# Classical Mechanics

Newton's First Law states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by an unbalanced force. This is also known as the law of inertia.

Newton's Second Law describes how force equals mass times acceleration, written as F = ma. When you push an object, it accelerates in the direction of the force. A heavier object requires more force to achieve the same acceleration.

Newton's Third Law states that for every action, there is an equal and opposite reaction. When you push against a wall, the wall pushes back against you with equal force.

Momentum is defined as mass times velocity, p = mv. In a closed system, total momentum is conserved. When two objects collide, their total momentum before equals their total momentum after.

Kinetic energy is the energy of motion, calculated as one-half times mass times velocity squared. A moving car has kinetic energy. A faster car has more kinetic energy. A heavier car also has more kinetic energy.

Potential energy is stored energy based on position. Gravitational potential energy equals mass times gravity times height. A ball held high has potential energy that converts to kinetic energy as it falls.

Work is done when a force moves an object through a distance. Work equals force times distance times the cosine of the angle between them. Lifting a box does positive work against gravity.

Power is the rate of doing work, measured in watts. One watt equals one joule per second. A 100-watt light bulb uses 100 joules of energy every second.

# Thermodynamics

Temperature measures the average kinetic energy of particles in a substance. Higher temperature means particles move faster. Absolute zero is the lowest possible temperature where all particle motion stops.

Heat is energy transferred due to temperature difference. Heat flows from hot objects to cold objects until thermal equilibrium is reached. This is described by the zeroth law of thermodynamics.

The first law of thermodynamics states that energy cannot be created or destroyed, only transformed. The total energy of an isolated system remains constant. This is conservation of energy.

The second law of thermodynamics states that entropy of an isolated system always increases. Heat flows spontaneously from hot to cold, never the reverse. Perpetual motion machines are impossible.

Entropy is a measure of disorder in a system. A shuffled deck has higher entropy than an ordered deck. The universe tends toward maximum entropy over time.

Specific heat capacity is the energy needed to raise one kilogram of a substance by one degree. Water has a high specific heat, which is why oceans moderate climate.

# Waves and Oscillations

A wave is a disturbance that transfers energy without transferring matter. Ocean waves transfer energy across the water, but the water molecules stay in roughly the same place.

Wavelength is the distance between consecutive wave peaks. Frequency is the number of waves passing a point per second. Wave speed equals wavelength times frequency.

Sound is a longitudinal wave where particles vibrate parallel to the direction of wave travel. Sound requires a medium like air or water to propagate. Sound cannot travel through vacuum.

Light is an electromagnetic wave that can travel through vacuum. The speed of light in vacuum is approximately 300 million meters per second. Nothing can travel faster than light.

Interference occurs when two waves meet. Constructive interference happens when peaks align, creating a larger wave. Destructive interference happens when a peak meets a trough, canceling out.

Resonance occurs when a system is driven at its natural frequency. A wine glass can shatter when a singer hits the right note. Bridges must be designed to avoid resonance with wind.

# Electricity and Magnetism

Electric charge comes in two types, positive and negative. Like charges repel, opposite charges attract. Electrons carry negative charge, protons carry positive charge.

Electric current is the flow of electric charge, measured in amperes. One ampere equals one coulomb of charge per second. Current flows from high potential to low potential.

Voltage is electric potential difference, measured in volts. A 9-volt battery has a potential difference of 9 volts between its terminals. Voltage drives current through a circuit.

Resistance opposes current flow, measured in ohms. Ohm's law states voltage equals current times resistance. A higher resistance means less current for the same voltage.

Power in a circuit equals voltage times current. A 100-watt device on 120 volts draws about 0.83 amperes. Power equals current squared times resistance.

Magnetic fields are produced by moving charges and magnets. A current-carrying wire creates a magnetic field around it. Earth's magnetic field protects us from solar radiation.

Electromagnetic induction occurs when a changing magnetic field creates an electric field. Generators use this to produce electricity. Transformers use induction to change voltage levels.

# Modern Physics

Einstein's special relativity shows that space and time are connected as spacetime. Time passes slower for fast-moving objects. Nothing can exceed the speed of light.

The famous equation E equals mc squared shows mass and energy are equivalent. A small amount of mass contains enormous energy. Nuclear reactions convert tiny mass differences into huge energy.

Quantum mechanics describes physics at atomic scales. Particles behave like waves and waves behave like particles. Electrons exist in probability clouds around nuclei.

The uncertainty principle states you cannot precisely know both position and momentum simultaneously. Measuring one disturbs the other. This is fundamental, not a measurement limitation.

Photons are particles of light with energy proportional to frequency. Higher frequency light has more energetic photons. Ultraviolet has more energy than infrared.

The photoelectric effect shows light behaves as particles. Light below a threshold frequency cannot eject electrons regardless of intensity. This proved light has particle nature.

Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons. Protons have positive charge, electrons have negative charge, neutrons are neutral.

Nuclear fission splits heavy atoms into lighter ones, releasing energy. Uranium-235 can undergo fission when struck by a neutron. Nuclear power plants use controlled fission.

Nuclear fusion combines light atoms into heavier ones, releasing energy. The sun produces energy by fusing hydrogen into helium. Fusion requires extremely high temperatures.

Radioactive decay is the spontaneous transformation of unstable nuclei. Half-life is the time for half the atoms to decay. Carbon-14 dating uses radioactive decay to determine age.

# Gravity and Orbits

Gravity is an attractive force between all masses. The gravitational force is proportional to both masses and inversely proportional to distance squared.

Orbits result from the balance between gravity and inertia. A satellite in orbit is constantly falling toward Earth but moving forward fast enough to keep missing.

Escape velocity is the minimum speed needed to escape a gravitational field. Earth's escape velocity is about 11 kilometers per second. Rockets must reach this speed to leave Earth.

Gravitational potential energy between two masses is negative and increases toward zero as distance increases. It takes energy to separate gravitationally bound objects.

Tides are caused by the differential gravitational pull of the Moon and Sun. The side of Earth closer to the Moon experiences stronger gravity than the far side.

General relativity describes gravity as the curvature of spacetime caused by mass. Massive objects bend the fabric of spacetime. Light follows curved paths near massive objects.

Black holes form when massive stars collapse. Their gravity is so strong that nothing, not even light, can escape once past the event horizon.

Gravitational waves are ripples in spacetime caused by accelerating masses. They were first detected in 2015 from merging black holes. This confirmed Einstein's prediction.

# Fluid Mechanics

Pressure is force per unit area. Atmospheric pressure at sea level is about 101,000 pascals. Pressure increases with depth in a fluid.

Buoyancy is the upward force on an object in a fluid. An object floats if its density is less than the fluid's density. Ships float because their average density is less than water.

Bernoulli's principle states that faster-moving fluid has lower pressure. Airplane wings use this to generate lift. The curved top surface causes air to move faster, reducing pressure.

Viscosity is a fluid's resistance to flow. Honey has high viscosity, water has low viscosity. Viscous forces dissipate energy in flowing fluids.

Turbulence is chaotic fluid flow at high speeds. Reynolds number predicts when flow becomes turbulent. Smooth flow is called laminar flow.

# Optics

Reflection occurs when light bounces off a surface. The angle of incidence equals the angle of reflection. Mirrors use reflection to form images.

Refraction is the bending of light when it passes between media. Light slows down in denser media and bends toward the normal. This is why pools appear shallower than they are.

Lenses use refraction to focus or spread light. Convex lenses converge light to a focal point. Concave lenses diverge light outward.

Total internal reflection occurs when light cannot escape a denser medium. This is how fiber optic cables transmit light. Diamonds sparkle due to total internal reflection.

The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays. They differ only in frequency and wavelength.

Color perception depends on wavelength of visible light. Red has the longest wavelength, violet the shortest. White light contains all colors mixed together.

Polarization describes the orientation of light wave oscillation. Polarized sunglasses block certain orientations, reducing glare. LCD screens use polarized light.

# Atomic and Nuclear Physics

Electrons occupy discrete energy levels in atoms. They can jump between levels by absorbing or emitting photons. Each element has a unique emission spectrum.

Isotopes are atoms of the same element with different numbers of neutrons. Carbon-12 and carbon-14 are isotopes. Some isotopes are stable, others are radioactive.

The strong nuclear force holds protons and neutrons together in the nucleus. It is much stronger than electromagnetic repulsion but only acts at very short range.

The weak nuclear force is responsible for radioactive beta decay. A neutron can transform into a proton by emitting an electron and antineutrino.

Particle accelerators speed up particles to study fundamental physics. The Large Hadron Collider discovered the Higgs boson. Higher energies probe smaller distance scales.

Antimatter consists of particles with opposite properties to normal matter. Positrons are antielectrons. When matter meets antimatter, they annihilate into pure energy.

# Astrophysics

Stars form from collapsing clouds of gas and dust. Gravity pulls material together until fusion ignites in the core. Our sun has been fusing hydrogen for 4.6 billion years.

Main sequence stars fuse hydrogen into helium in their cores. The outward pressure from fusion balances inward gravitational pull. Stars spend most of their lives on the main sequence.

Red giants form when stars exhaust their core hydrogen. The core contracts and heats while outer layers expand and cool. Our sun will become a red giant in about 5 billion years.

Supernovae are explosive deaths of massive stars. The core collapses suddenly, and the rebounding material creates heavy elements. Many elements in your body came from supernovae.

Neutron stars are incredibly dense remnants of supernovae. A teaspoon of neutron star material would weigh billions of tons. Pulsars are rapidly rotating neutron stars.

White dwarfs are Earth-sized stellar remnants supported by electron degeneracy pressure. They slowly cool over billions of years. Our sun will end as a white dwarf.

The Hertzsprung-Russell diagram plots stars by temperature and luminosity. Main sequence stars form a diagonal band. Giants and white dwarfs occupy other regions.

Galaxies are vast collections of stars, gas, and dark matter. The Milky Way contains hundreds of billions of stars. Galaxies cluster together under mutual gravity.

Dark matter is invisible matter detected only through its gravitational effects. It makes up about 27 percent of the universe. Its nature remains unknown.

Dark energy is causing the expansion of the universe to accelerate. It makes up about 68 percent of the universe. Its nature is even more mysterious than dark matter.

The Big Bang theory describes the origin of the universe from an extremely hot, dense state about 13.8 billion years ago. The universe has been expanding ever since.

Cosmic microwave background radiation is the afterglow of the Big Bang. It fills all of space at a temperature of 2.7 kelvin. Its patterns reveal the early universe's structure.

'''
    
    # Repeat the content to make it larger (aim for ~1MB)
    full_text = physics_text * 8  # ~1MB
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Generated physics dataset: {filepath}")
    print(f"Size: {len(full_text):,} characters")


def setup_physics_dataset():
    """Generate the physics dataset for training."""
    os.makedirs('data', exist_ok=True)
    filepath = 'data/input.txt'
    
    print("\n" + "="*70)
    print(" Physics Dataset Setup")
    print("="*70)
    print("\nGenerating physics concepts dataset...")
    print("Topics include: Classical Mechanics, Thermodynamics, Waves,")
    print("Electricity & Magnetism, Modern Physics, Astrophysics")
    
    generate_physics_dataset(filepath)
    
    print(f"\n✅ Physics dataset ready at: {filepath}")
    print("\nExample prompts to try after training:")
    print("  python generate.py -p 'Newton' -n 200")
    print("  python generate.py -p 'Energy' -n 200")
    print("  python generate.py -p 'The force' -n 200")
    print("  python generate.py -p 'Quantum' -n 200")
    print("\nWhat to look for in output:")
    print("  Physics terms, equations described in words, scientific concepts")


def main():
    parser = argparse.ArgumentParser(description='Generate physics training dataset')
    parser.add_argument('--info', action='store_true', help='Show dataset info')
    
    args = parser.parse_args()
    
    if args.info:
        print("\nPhysics Dataset Information:")
        print("-" * 40)
        print("Size: ~1MB of physics text")
        print("Topics: Classical Mechanics, Thermodynamics, Waves,")
        print("        Electricity & Magnetism, Modern Physics, Astrophysics")
        print("\nExample prompts: 'Newton', 'Energy', 'Gravity', 'Quantum'")
    else:
        setup_physics_dataset()


if __name__ == "__main__":
    main()

