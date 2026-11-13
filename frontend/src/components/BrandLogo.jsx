function BrandLogo({ compact = false }) {
  return (
    <span
      className={`relative block shrink-0 overflow-hidden ${compact ? 'h-10 w-[4.75rem]' : 'h-14 w-[6.75rem]'}`}
    >
      <img
        src="/retailiq-logo.png"
        alt="RetailIQ"
        className={`absolute max-w-none ${
          compact
            ? 'left-[-2.1rem] top-[-1.5rem] w-[6.9rem]'
            : 'left-[-2.75rem] top-[-2rem] w-[9.05rem]'
        }`}
      />
    </span>
  )
}

export default BrandLogo
